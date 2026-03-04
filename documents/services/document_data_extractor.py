from datetime import datetime
from decimal import Decimal, InvalidOperation
import enum
import json
import re
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from pdf2image import convert_from_bytes
import base64
from openai import AzureOpenAI
from thefuzz import fuzz
import io
from typing import Dict, List, Tuple, TypeVar, Optional, Any
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult, DocumentContentFormat, DocumentAnalysisFeature
from shared.confidence.openai_confidence import evaluate_confidence as evaluate_confidence_openai
from shared.confidence.document_intelligence_confidence import SearchContext, evaluate_confidence as evaluate_confidence_di
import logging

logger = logging.getLogger(__name__)

LOW_CONFIDENCE_THRESHOLD = 0.7

class ReportType(enum.Enum):
    INDIVIDUAL = "INDIVIDUAL"
    COMPANY = "COMPANY"

ResponseFormatT = TypeVar(
    "ResponseFormatT"
)


INDIVIDUAL_REQUIRED_FIELDS = [
    'repayment_to_banks',
    'utilisation',
    'special_attention_accounts',
    'legal_cases',
    'blacklist',
]

COMPANY_REQUIRED_FIELDS = [
    'repayment_to_banks',
    'utilisation',
    'special_attention_accounts',
    'legal_cases',
    'blacklist',
    'years_in_business',
    'type_of_company',
    'nature_of_business',
    'number_of_directors_or_partners',
    'paid_up_capital',
    'financial_report_date',
    'turnover',
    'net_profit',
    'retained_profit',
    'net_worth',
    'net_current_assets',
    'current_ratio',
    'gearing_ratio',
]

class DocumentDataExtractorOptions:
    """Defines the configuration options for extracting data from a document using Azure OpenAI."""

    def __init__(self, page_start: Optional[int], page_end: Optional[int], doc_intelligence_endpoint: str, openai_endpoint: str, deployment_name: str, max_tokens: int = 4096, temperature: float = 0.1, top_p: float = 0.1):
        """Initializes a new instance of the DocumentDataExtractorOptions class.

        :param page_start: The starting page number of the document to extract data from.
        :param page_end: The ending page number of the document to extract data from.
        :param doc_intelligence_endpoint: The Azure Document Intelligence endpoint to use for the request.
        :param openai_endpoint: The Azure OpenAI endpoint to use for the request.
        :param deployment_name: The name of the model deployment to use for the request.
        :param max_tokens: The maximum number of tokens to generate in the response. Default is 4096.
        :param temperature: The sampling temperature for the model. Default is 0.1.
        :param top_p: The nucleus sampling parameter for the model. Default is 0.1.
        """

        self.system_prompt = f"""You are an AI assistant that extracts data from specific tables in documents. You will be provided with the markdown content of the document which includes text and tables."""
        self.page_start = page_start
        self.page_end = page_end
        self.openai_endpoint = openai_endpoint
        self.doc_intelligence_endpoint = doc_intelligence_endpoint
        self.deployment_name = deployment_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p


class DocumentDataExtractor:
    """Defines a class for extracting structured data from a document using Azure OpenAI GPT models that support image inputs."""

    def __init__(self, credential: DefaultAzureCredential):
        """Initializes a new instance of the DocumentDataExtractor class.

        :param credential: The Azure credential to use for authenticating with the Azure OpenAI service.
        """

        self.credential = credential
        self.result: AnalyzeResult = None
        self.report_type: ReportType = None
        self.relevant_values: Dict = {}
        self.options: DocumentDataExtractorOptions = None
        self.bytes: bytes = None
        self.table_page_ranges: Dict[str, Tuple[int, int]] = {}
        self.di_confidence: Dict = {}
        self.di_conduct = None

    def __safe_get_cell__(self, table, r_idx: int, c_idx: int) -> Optional[str]:
        """Safely gets and strips a cell value from a table row, returning None if the cell doesn't exist."""
        value = table[r_idx].get(c_idx)
        if value is None:
            logger.debug("Missing cell at row %d, col %d", r_idx, c_idx)
            return None
        return value.strip()

    def extract_using_doc_intelligence(self, document_bytes: bytes, options: DocumentDataExtractorOptions): 
        logger.info("Starting extraction, document size: %d bytes", len(document_bytes))
        
        try:
            self.options = options
            self.bytes = document_bytes
            di_client = self.__get_document_intelligence_client__(options)
        except Exception as e:
            logger.error("Failed to create Document Intelligence client: %s", e, exc_info=True)
            raise

        if options.page_start and options.page_end:
            page_range = f"{options.page_start}-{options.page_end}"
        else:
            page_range = None
        
        try:
            poller = di_client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=document_bytes,
                pages=page_range,
                output_content_format=DocumentContentFormat.MARKDOWN,
                content_type="application/pdf",
                features=[DocumentAnalysisFeature.OCR_HIGH_RESOLUTION]
            )

            self.result: AnalyzeResult = poller.result()
            logger.info("Document Intelligence returned %d tables, %d paragraphs", len(self.result.tables or []), len(self.result.paragraphs or []))

            self.report_type = self.__classify_report_type__()
            logger.info("Classified report type as: %s", self.report_type.value)

            tagged_tables = self.__identify_tables_from_json__()
            logger.info("Tagged %d relevant tables for extraction", len(tagged_tables))

            extracted_data = self.__extract_from_tagged_tables__(tagged_tables)
            logger.info("Extracted data: %s", extracted_data)

            confidence_di = evaluate_confidence_di(
                extract_result=self.di_confidence,
                analyze_result=self.result
            )

            logger.info("Document Intelligence confidence evaluation: %s", confidence_di)

            parsed_data, flags = self.__run_extraction_pipeline__(extracted_data, confidence_di)
            logger.info("Parsed extracted data: %s", parsed_data)

            mapped_data = self.__map_parsed_data__(parsed_data)
            mapped_data['_flags'] = flags
            logger.info("Completed extraction successfully")

        except Exception as e:
            logger.error("Extraction failed: %s", e, exc_info=True)
            raise
        
        # Convert Decimal values to float for JSON serialization
        result = {}
        for k, v in mapped_data.items():
            if isinstance(v, Decimal):
                result[k] = float(v)
            else:
                result[k] = v
        return result
    
    def __to_decimal__(self, value) -> Decimal:
        """Converts a value to Decimal. Skips string normalization if value is already numeric."""
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        return self.__str_to_decimal__(self.__normalize_numeric_str__(value))

    def __normalize_numeric_str__(self, value: str) -> str:
        """Normalizes a numeric string to handle OCR errors.
        Handles cases where:
        - Commas are used as thousand separators: '2,091,202.00' -> '2091202.00'
        - Commas are misread as periods: '2,091.202.00' -> '2091202.00'
        - Comma is used as decimal separator (European format): '1,22' -> '1.22'
        """
        # Handle nil/dash values
        stripped = value.strip()
        if stripped == '-' or stripped == '–' or stripped == '—' or stripped == '':
            return '0'
        
        # Remove all commas: commas are either thousand separators or OCR errors
        stripped = stripped.replace(',', '')
        
        parts = stripped.split('.')
        if len(parts) <= 2:
            return stripped
        # Multiple periods: all but last period are commas since numbers are in 2 decimal places
        return ''.join(parts[:-1]) + '.' + parts[-1]

    def __get_page__(self, bounding_regions) -> Optional[Tuple[int, int]]:
        """Gets the min and max page numbers from bounding regions."""
        pages = set()
        for region in (bounding_regions or []):
            pages.add(region.page_number)
        if not pages:
            return
        return min(pages), max(pages)

    def __record_table_pages__(self, table_idx: int, table_tag: str):
        """Records the page range for a tagged table type."""
        table = self.result.tables[table_idx]
        min_page, max_page = self.__get_page__(table.bounding_regions)
        # Expand existing range if this tag was already seen (e.g. multi-page CCRIS_DETAILS_MULTI_MID tables)
        if table_tag in self.table_page_ranges:
            existing_min, existing_max = self.table_page_ranges[table_tag]
            min_page = min(existing_min, min_page)
            max_page = max(existing_max, max_page)
        self.table_page_ranges[table_tag] = (min_page, max_page)

    def __get_required_fields__(self) -> List[str]:
        """Returns the list of required parsed-data keys for the current report type."""
        if self.report_type == ReportType.INDIVIDUAL:
            return list(INDIVIDUAL_REQUIRED_FIELDS)
        else:
            base = list(COMPANY_REQUIRED_FIELDS)
            # Partnership reports don't need most financial fields
            if self.relevant_values.get('partnership') is not None:
                for key in ['paid_up_capital', 'financial_report_date', 'turnover', 'net_profit',
                            'retained_profit', 'net_worth', 'net_current_assets', 'current_ratio', 'gearing_ratio']:
                    if key in base:
                        base.remove(key)
            return base

    def __run_extraction_pipeline__(self, extracted_data: Dict, confidence_di: Dict) -> Tuple[Dict, Dict]:
        """
        Orchestrates the three-layer extraction:
          Layer 0: Azure Document Intelligence JSON table extraction, passed as extracted_data.
          Layer 1: Markdown-only extraction for essential cross-checking with layer 0.
          Layer 2: Selective fallback using Markdown + Image.

        Returns (parsed_data, flags) where flags contains missing/low-confidence info.
        """
        flags: Dict[str, List[str]] = {
            'missing': [],
            'low_confidence': [],
            'validation_failed': [],
        }

        # ---- Layer 0: Parse DI data and validate ----
        logger.info("=== Layer 0: Document Intelligence table extraction ===")
        parsed_data = self.__parse_extracted_data_layer0__(extracted_data)
        validation_l0 = self.__validate_parsed_data__(parsed_data, extracted_data)
        missing_l0, low_conf_l0 = self.__identify_issues__(parsed_data, confidence_di)

        logger.info("Layer 0 missing fields: %s", missing_l0)
        logger.info("Layer 0 low confidence fields: %s", low_conf_l0)
        logger.info("Layer 0 validation failures: %s", list(validation_l0.keys()))

        # ---- Layer 1: Markdown ----
        logger.info("=== Layer 1: Markdown-only GPT-4o for cross-checking all fields ===")
        markdown_prompt = self.__get_markdown_prompt__()
        markdown_result, markdown_choice = self.extract_using_markdown_with_confidence(markdown_prompt)

        confidence_l1 = {}
        if markdown_result and markdown_choice:
            confidence_l1 = evaluate_confidence_openai(
                extract_result=markdown_result,
                choice=markdown_choice
            )
            logger.info("Layer 1 OpenAI confidence: %s", confidence_l1)

            # Compare markdown results with DI results for all fields.
            # If the fields are the same, and are both above the confidence threshold,
            # then we keep the DI value. Otherwise, mark for Layer 2 fallback.
            fields_needing_fallback_l2 = self.__compare_markdown_results_with_di__(
                markdown_result, parsed_data, extracted_data, confidence_di, confidence_l1
            )
        else:
            logger.warning("Layer 1: Markdown extraction returned no result, falling back on DI issues")
            # Without markdown cross-check, fall back to DI-only issue detection
            fields_needing_fallback_l2 = set(missing_l0) | set(low_conf_l0) | set(validation_l0.keys())

        # Re-validate after any Layer 1 adoptions
        validation_l1 = self.__validate_parsed_data__(parsed_data, extracted_data)
        missing_l1, low_conf_l1 = self.__identify_issues__(parsed_data, confidence_l1 if confidence_l1 else confidence_di)
        logger.info("Post-Layer 1 missing fields: %s", missing_l1)
        logger.info("Post-Layer 1 low confidence fields: %s", low_conf_l1)
        logger.info("Post-Layer 1 validation failures: %s", list(validation_l1.keys()))

        # Merge any newly discovered issues into fields needing fallback
        fields_needing_fallback_l2 = fields_needing_fallback_l2 | set(missing_l1) | set(validation_l1.keys())

        # Remove fields that are already N/A (intentionally absent, e.g. partnership financials)
        fields_needing_fallback_l2 = {
            f for f in fields_needing_fallback_l2
            if parsed_data.get(f) != 'N/A'
        }

        # Force additional validation of number of non-zeroes using image + markdown from GPT-4o.
        if self.di_conduct is not None and 'repayment_to_banks' not in fields_needing_fallback_l2:
            fields_needing_fallback_l2.add('repayment_to_banks')

        if not fields_needing_fallback_l2:
            logger.info("All fields resolved at Layer 1")
            return parsed_data, flags

        # ---- Layer 2: Markdown + Image fallback ----
        logger.info("=== Layer 2: Markdown + Image fallback for fields: %s ===", fields_needing_fallback_l2)
        self.__run_layer2_image_fallback__(extracted_data, parsed_data, fields_needing_fallback_l2, markdown_result)

        # Final validation
        validation_l2 = self.__validate_parsed_data__(parsed_data, extracted_data)
        required_fields = self.__get_required_fields__()

        # Build final flags
        for field in required_fields:
            if parsed_data.get(field) is None:
                flags['missing'].append(field)
            elif field in validation_l2:
                flags['validation_failed'].append(field)

        # Check final confidence from all layers
        # We only flag low confidence for fields that are present but weren't caught by validation
        all_confidence = {}
        if confidence_di:
            all_confidence.update(confidence_di)
        if confidence_l1:
            all_confidence.update(confidence_l1)

        for field in required_fields:
            if field in flags['missing'] or field in flags['validation_failed']:
                continue
            if parsed_data.get(field) is not None:
                field_conf = self.__get_field_confidence__(field, all_confidence)
                if field_conf is not None and field_conf < LOW_CONFIDENCE_THRESHOLD:
                    flags['low_confidence'].append(field)

        if flags['missing']:
            logger.warning("FINAL - Missing fields: %s", flags['missing'])
        if flags['low_confidence']:
            logger.warning("FINAL - Low confidence fields: %s", flags['low_confidence'])
        if flags['validation_failed']:
            logger.warning("FINAL - Validation failed fields: %s", flags['validation_failed'])

        return parsed_data, flags
    
    def __get_field_confidence__(self, field: str, confidence: Dict) -> Optional[float]:
        """Extracts the confidence score for a field from a confidence dict, e.g. nested {'confidence': float, 'value': ...} or just {'confidence': float}."""
        if field not in confidence:
            return None
        val = confidence[field]
        if isinstance(val, dict) and 'confidence' in val:
            return val['confidence']
        if isinstance(val, (int, float)):
            return val
        return None
    
    def __identify_issues__(self, parsed_data: Dict, confidence: Dict) -> Tuple[List[str], List[str]]:
        """Returns (missing_fields, low_confidence_fields) for current state."""
        required = self.__get_required_fields__()
        missing = []
        low_conf = []

        for field in required:
            val = parsed_data.get(field)
            if val is None:
                missing.append(field)
                continue
            # Check if N/A (intentionally absent)
            if val == 'N/A':
                continue
            # Check confidence
            field_conf = self.__get_field_confidence__(field, confidence)
            if field_conf is not None and field_conf < LOW_CONFIDENCE_THRESHOLD:
                low_conf.append(field)

        return missing, low_conf
    
    def __validate_parsed_data__(self, parsed_data: Dict, extracted_data: Dict) -> Dict[str, str]:
        """
        Performs validation checks on parsed data. Returns a dict of field_name -> failure_reason for fields that fail validation.
        """
        failures = {}

        # --- Utilisation cross-check: total_outstanding_balance_0 == _1, total_limit_0 == _1 ---
        util_keys_0 = ['total_outstanding_balance_0', 'total_outstanding_balance_1', 'total_limit_0', 'total_limit_1']
        if all(extracted_data.get(k) is not None for k in util_keys_0):
            if (self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_0']) !=
                    self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_1'])):
                failures['utilisation'] = 'total_outstanding_balance mismatch'
            if (self.__normalize_numeric_str__(extracted_data['total_limit_0']) !=
                    self.__normalize_numeric_str__(extracted_data['total_limit_1'])):
                failures['utilisation'] = 'total_limit mismatch'

        # --- Special attention accounts cross-check (individual) ---
        if self.report_type == ReportType.INDIVIDUAL:
            spa_keys = ['special_attention_accounts_0', 'special_attention_accounts_1']
            if all(extracted_data.get(k) is not None for k in spa_keys):
                str0 = extracted_data['special_attention_accounts_0']
                str1 = extracted_data['special_attention_accounts_1']
                if str0[0].upper() != str1[0].upper():
                    failures['special_attention_accounts'] = 'SAA value mismatch'

        # --- Legal cases sum check ---
        if self.report_type == ReportType.INDIVIDUAL:
            legal_keys = ['legal_non_personal', 'legal_personal']
        else:
            legal_keys = ['legal_non_personal_entity', 'legal_personal_entity']

        if all(extracted_data.get(k) is not None for k in legal_keys):
            try:
                np_val = int(extracted_data[legal_keys[0]])
                p_val = int(extracted_data[legal_keys[1]])
                calc = np_val + p_val
                if extracted_data.get('legal_cases_count') is not None:
                    if calc != extracted_data['legal_cases_count']:
                        failures['legal_cases'] = f'legal sum {calc} != legal_cases_count {extracted_data["legal_cases_count"]}'
            except (ValueError, TypeError):
                failures['legal_cases'] = 'Could not parse legal case counts'

        # --- Financial statements validation (non-partnership company only) ---
        if self.report_type == ReportType.COMPANY and self.relevant_values.get('partnership') is None:
            # Revenue cross-check
            if extracted_data.get('revenue_0') is not None and extracted_data.get('revenue_1') is not None:
                if (self.__normalize_numeric_str__(extracted_data['revenue_0']) !=
                        self.__normalize_numeric_str__(extracted_data['revenue_1'])):
                    if not (self.__normalize_numeric_str__(extracted_data['revenue_0']) == '0' and
                            self.__normalize_numeric_str__(extracted_data['revenue_1']) == '0'):
                        failures['turnover'] = 'revenue mismatch between financials_and_shareholders and financial_statements'

            # Profit after tax cross-check
            if extracted_data.get('profit_after_tax_0') is not None and extracted_data.get('profit_after_tax_1') is not None:
                if (self.__normalize_numeric_str__(extracted_data['profit_after_tax_0']) !=
                        self.__normalize_numeric_str__(extracted_data['profit_after_tax_1'])):
                    if not (self.__normalize_numeric_str__(extracted_data['profit_after_tax_0']) == '0' and
                            self.__normalize_numeric_str__(extracted_data['profit_after_tax_1']) == '0'):
                        failures['net_profit'] = 'profit_after_tax mismatch between financials_and_shareholders and financial_statements'

            # Balance sheet sum check: TA == NCA + CA, TL == NCL + CL + LTL
            fs_keys = ['current_assets', 'current_liabilities', 'non_current_assets', 'total_assets',
                       'non_current_liabilities', 'long_term_liabilities', 'total_liabilities']
            if all(extracted_data.get(k) is not None for k in fs_keys):
                nca = self.__to_decimal__(extracted_data['non_current_assets'])
                ca = self.__to_decimal__(extracted_data['current_assets'])
                ta = self.__to_decimal__(extracted_data['total_assets'])
                ncl = self.__to_decimal__(extracted_data['non_current_liabilities'])
                cl = self.__to_decimal__(extracted_data['current_liabilities'])
                ltl = self.__to_decimal__(extracted_data['long_term_liabilities'])
                tl = self.__to_decimal__(extracted_data['total_liabilities'])
                if ta != nca + ca:
                    failures['net_current_assets'] = f'total_assets {ta} != non_current_assets {nca} + current_assets {ca}'
                if tl != ncl + cl + ltl:
                    failures['net_current_assets'] = f'total_liabilities {tl} != sum of components'

            # Current ratio validation
            if extracted_data.get('current_ratio') is not None and extracted_data.get('current_assets') is not None and extracted_data.get('current_liabilities') is not None:
                ca = self.__to_decimal__(extracted_data['current_assets'])
                cl = self.__to_decimal__(extracted_data['current_liabilities'])
                extracted_cr = self.__to_decimal__(extracted_data['current_ratio'])
                if cl > 0:
                    cr = ca / cl
                    if abs(cr - extracted_cr) >= Decimal('0.01'):
                        failures['current_ratio'] = f'current_ratio {extracted_cr} != calculated {cr}'

            # Gearing ratio validation
            if all(extracted_data.get(k) is not None for k in ['gearing_ratio', 'debt_to_equity_ratio', 'net_worth', 'total_liabilities']):
                tl = self.__to_decimal__(extracted_data['total_liabilities'])
                nw = self.__to_decimal__(extracted_data['net_worth'])
                extracted_gr = self.__to_decimal__(extracted_data['gearing_ratio'])
                extracted_der = self.__to_decimal__(extracted_data['debt_to_equity_ratio'])
                if nw > 0:
                    calc_gr = tl / nw
                    if abs(extracted_gr - extracted_der) >= Decimal('0.01'):
                        failures['gearing_ratio'] = 'gearing_ratio and debt_to_equity_ratio mismatch'
                    elif abs(extracted_gr - calc_gr) >= Decimal('0.01'):
                        failures['gearing_ratio'] = f'gearing_ratio {extracted_gr} != calculated {calc_gr}'

        return failures
    
    def __parse_extracted_data_layer0__(self, extracted_data: Dict) -> Dict:
        """Layer 0: Parse Document Intelligence-extracted table data into parsed fields.
        """
        parsed_data = {}

        if self.report_type == ReportType.INDIVIDUAL:
            # CCRIS conduct from DI
            if extracted_data.get('ccris_conduct'):
                parsed_data['repayment_to_banks'] = self.__parse_conduct_values__(extracted_data['ccris_conduct'])
                self.di_conduct = extracted_data['ccris_conduct']

            # Utilisation
            util_keys = ['total_outstanding_balance_0', 'total_outstanding_balance_1', 'total_limit_0', 'total_limit_1']
            if all(extracted_data.get(key) is not None for key in util_keys):
                if (self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_0']) ==
                        self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_1']) and
                        self.__normalize_numeric_str__(extracted_data['total_limit_0']) ==
                        self.__normalize_numeric_str__(extracted_data['total_limit_1'])):
                    bal = self.__to_decimal__(extracted_data['total_outstanding_balance_0'])
                    limit = self.__to_decimal__(extracted_data['total_limit_0'])
                    if limit > 0:
                        parsed_data['utilisation'] = bal / limit * 100

            # Special attention accounts
            spa_keys = ['special_attention_accounts_0', 'special_attention_accounts_1']
            if all(extracted_data.get(key) is not None for key in spa_keys):
                # saa_0 is 'NO', saa_1 is 'N'
                str0 = extracted_data['special_attention_accounts_0']
                str1 = extracted_data['special_attention_accounts_1']
                if str0[0] == str1[0]:
                    parsed_data['special_attention_accounts'] = str0

            # Legal cases
            legal_keys = ['legal_non_personal', 'legal_personal']
            if all(extracted_data.get(key) is not None for key in legal_keys):
                if extracted_data['legal_non_personal'] == '0' and extracted_data['legal_personal'] == '0' and self.relevant_values.get('legal_cases') is None:
                    parsed_data['legal_cases'] = 0
                else:
                    try:
                        np_val = int(extracted_data['legal_non_personal'])
                        p_val = int(extracted_data['legal_personal'])
                        calc = np_val + p_val
                        if extracted_data.get('legal_cases_count') is not None:
                            if calc == extracted_data['legal_cases_count']:
                                parsed_data['legal_cases'] = calc
                    except (ValueError, TypeError):
                        pass

            # Blacklist (trade reference)
            if self.relevant_values.get('trade_reference') is not None:
                if extracted_data.get('trade_reference_count') is not None:
                    parsed_data['blacklist'] = extracted_data['trade_reference_count']
            elif self.relevant_values.get('trade_reference') is None:
                parsed_data['blacklist'] = 0

        elif self.report_type == ReportType.COMPANY:
            # N/A when no CCRIS data at all
            util_keys = ['total_outstanding_balance_0', 'total_outstanding_balance_1', 'total_limit_0', 'total_limit_1']
            if all(extracted_data.get(key) is None for key in util_keys) and extracted_data.get('ccris_conduct') is None:
                parsed_data['repayment_to_banks'] = 'N/A'
                parsed_data['utilisation'] = 'N/A'
            else:
                # CCRIS conduct from DI
                if extracted_data.get('ccris_conduct'):
                    parsed_data['repayment_to_banks'] = self.__parse_conduct_values__(extracted_data['ccris_conduct'])
                    self.di_conduct = extracted_data['ccris_conduct']

                # Utilisation
                if all(extracted_data.get(key) is not None for key in util_keys):
                    if (self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_0']) ==
                            self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_1']) and
                            self.__normalize_numeric_str__(extracted_data['total_limit_0']) ==
                            self.__normalize_numeric_str__(extracted_data['total_limit_1'])):
                        bal = self.__to_decimal__(extracted_data['total_outstanding_balance_0'])
                        limit = self.__to_decimal__(extracted_data['total_limit_0'])
                        if limit > 0:
                            parsed_data['utilisation'] = bal / limit * 100

            # Special attention accounts (company uses entity)
            if extracted_data.get('special_attention_accounts_entity') is not None:
                parsed_data['special_attention_accounts'] = extracted_data['special_attention_accounts_entity']

            # Legal cases
            legal_keys = ['legal_non_personal_entity', 'legal_personal_entity']
            if all(extracted_data.get(key) is not None for key in legal_keys):
                if extracted_data['legal_non_personal_entity'] == '0' and extracted_data['legal_personal_entity'] == '0' and self.relevant_values.get('legal_cases') is None:
                    parsed_data['legal_cases'] = 0
                else:
                    try:
                        np_val = int(extracted_data['legal_non_personal_entity'])
                        p_val = int(extracted_data['legal_personal_entity'])
                        calc = np_val + p_val
                        if extracted_data.get('legal_cases_count') is not None:
                            if calc == extracted_data['legal_cases_count']:
                                parsed_data['legal_cases'] = calc
                    except (ValueError, TypeError):
                        pass

            # Blacklist (trade reference)
            if self.relevant_values.get('trade_reference') is not None:
                if extracted_data.get('trade_reference_count') is not None:
                    parsed_data['blacklist'] = extracted_data['trade_reference_count']
            if self.relevant_values.get('trade_reference') is None:
                parsed_data['blacklist'] = 0

            # Snapshot
            if extracted_data.get('registration_date') is not None:
                parsed_data['years_in_business'] = self.__calculate_years__(extracted_data['registration_date'])
            if extracted_data.get('type') is not None:
                type_val = extracted_data['type']
                if self.__is_fuzzy_match__(type_val, 'limited by shares private limited'):
                    parsed_data['type_of_company'] = 'Sdn Bhd'
                else:
                    parsed_data['type_of_company'] = 'Non - Sdn Bhd'
            if extracted_data.get('msic') is not None:
                parsed_data['nature_of_business'] = extracted_data['msic']

            # Partnership handling
            if self.relevant_values.get('partnership') is not None and parsed_data.get('type_of_company') == 'Non - Sdn Bhd':
                # partnership form does not need paid_up_capital
                #parsed_data['paid_up_capital'] = 'N/A'
                parsed_data['financial_report_date'] = 'N/A'
                parsed_data['turnover'] = 'N/A'
                parsed_data['net_profit'] = 'N/A'
                parsed_data['retained_profit'] = 'N/A'
                parsed_data['net_worth'] = 'N/A'
                parsed_data['net_current_assets'] = 'N/A'
                parsed_data['current_ratio'] = 'N/A'
                parsed_data['gearing_ratio'] = 'N/A'
                if extracted_data.get('partner_count') is not None:
                    parsed_data['number_of_directors_or_partners'] = extracted_data['partner_count']
            else:
                if extracted_data.get('director_count') is not None:
                    parsed_data['number_of_directors_or_partners'] = extracted_data['director_count']
                if extracted_data.get('paid_up_capital') is not None:
                    parsed_data['paid_up_capital'] = self.__to_decimal__(extracted_data['paid_up_capital'])
                if extracted_data.get('financial_year_end') is not None:
                    parsed_data['financial_report_date'] = self.__reformat_date__(extracted_data['financial_year_end'])

                # Revenue cross-check
                if extracted_data.get('revenue_0') is not None and extracted_data.get('revenue_1') is not None:
                    if (self.__normalize_numeric_str__(extracted_data['revenue_0']) ==
                            self.__normalize_numeric_str__(extracted_data['revenue_1'])):
                        parsed_data['turnover'] = self.__to_decimal__(extracted_data['revenue_0'])
                    elif (self.__normalize_numeric_str__(extracted_data['revenue_0']) == '0' and
                          self.__normalize_numeric_str__(extracted_data['revenue_1']) == '0'):
                        parsed_data['turnover'] = Decimal(0)

                # Profit after tax cross-check
                if extracted_data.get('profit_after_tax_0') is not None and extracted_data.get('profit_after_tax_1') is not None:
                    if (self.__normalize_numeric_str__(extracted_data['profit_after_tax_0']) ==
                            self.__normalize_numeric_str__(extracted_data['profit_after_tax_1'])):
                        parsed_data['net_profit'] = self.__to_decimal__(extracted_data['profit_after_tax_0'])
                    elif (self.__normalize_numeric_str__(extracted_data['profit_after_tax_0']) == '0' and
                          self.__normalize_numeric_str__(extracted_data['profit_after_tax_1']) == '0'):
                        parsed_data['net_profit'] = Decimal(0)

                if extracted_data.get('retained_earning') is not None:
                    parsed_data['retained_profit'] = self.__to_decimal__(extracted_data['retained_earning'])
                if extracted_data.get('net_worth') is not None:
                    parsed_data['net_worth'] = self.__to_decimal__(extracted_data['net_worth'])

                # Net current assets
                fs_keys = ['current_assets', 'current_liabilities', 'non_current_assets', 'total_assets',
                           'non_current_liabilities', 'long_term_liabilities', 'total_liabilities']
                if all(extracted_data.get(key) is not None for key in fs_keys):
                    nca = self.__to_decimal__(extracted_data['non_current_assets'])
                    ca = self.__to_decimal__(extracted_data['current_assets'])
                    ta = self.__to_decimal__(extracted_data['total_assets'])
                    ncl = self.__to_decimal__(extracted_data['non_current_liabilities'])
                    cl = self.__to_decimal__(extracted_data['current_liabilities'])
                    ltl = self.__to_decimal__(extracted_data['long_term_liabilities'])
                    tl = self.__to_decimal__(extracted_data['total_liabilities'])
                    valid_ca_cl = (ta == nca + ca) and (tl == ncl + cl + ltl)
                    if valid_ca_cl:
                        parsed_data['net_current_assets'] = ca - cl
                        if extracted_data.get('current_ratio') is not None:
                            extracted_cr = self.__to_decimal__(extracted_data['current_ratio'])
                            cr = ca / cl if cl > 0 else Decimal(0)
                            if abs(cr - extracted_cr) < Decimal('0.01'):
                                parsed_data['current_ratio'] = extracted_cr

                # Gearing ratio
                bal_keys = ['gearing_ratio', 'debt_to_equity_ratio', 'net_worth', 'total_liabilities']
                if all(extracted_data.get(key) is not None for key in bal_keys):
                    tl = self.__to_decimal__(extracted_data['total_liabilities'])
                    nw = self.__to_decimal__(extracted_data['net_worth'])
                    calculated_gr = tl / nw if nw > 0 else Decimal(0)
                    extracted_gr = self.__to_decimal__(extracted_data['gearing_ratio'])
                    extracted_der = self.__to_decimal__(extracted_data['debt_to_equity_ratio'])
                    valid_gr = (abs(extracted_gr - extracted_der) < Decimal('0.01')) and (abs(extracted_gr - calculated_gr) < Decimal('0.01'))
                    if valid_gr:
                        parsed_data['gearing_ratio'] = extracted_gr

        return parsed_data
    
    def __get_markdown_prompt__(self) -> str:
        """Markdown prompt asking GPT-4o to:
        - extract all fields (depending on report type) so that each field can be compared against the fields extracted from document intelligence table
        - verify that for the sections 'd1: legal cases (subject as defendant)' and 'd2: legal cases (subject as plaintiff)', there is either 'no information available' below these section headings, or there are tables associated with these sections. If there are tables, count the number of legal cases in the summary table (to be compared against the legal cases count extracted from document intelligence).
        - verify that for the section 'e2: trade reference', there is either 'no information available' below this section heading, or there are tables associated with this section with the subheadings 'the following information are in relation to account no' or 'aging information'. If there are tables, count the number of trade references in the summary table (to be compared against the trade reference count extracted from document intelligence).
        """
        common_instructions = (
            "You are given the markdown content of a credit report document. "
            "Extract the following fields from the document. "
            "If the value is 0, it may be represented as a dash '-' or an en-dash '–' or an em-dash '—'. "
            "If the value is 0.00, return 0.00 not null. "
            "Brackets surrounding a numerical value indicates that the numerical value is negative. "
            "If any field is not present in the document, return null for that field. "
        )

        # Legal cases verification instructions (common to both report types)
        legal_instructions = (
            "For the sections 'D1: LEGAL CASES (SUBJECT AS DEFENDANT)' and 'D2: LEGAL CASES (SUBJECT AS PLAINTIFF)', "
            "check if there is 'No Information Available' below each section heading. "
            "If 'No Information Available' appears, the legal case count for that section is 0. "
            "If there are tables under these sections, count the total number of distinct legal case rows in the summary tables. "
            "Return the total count of legal cases across both sections as 'legal_cases_count'. "
        )

        # Trade reference verification instructions (common to both report types)
        trade_ref_instructions = (
            "For the section 'E2: TRADE REFERENCE', "
            "check if there is 'No Information Available' below the section heading. "
            "If 'No Information Available' appears, return false for 'has_trade_reference'. "
            "If there are tables under this section with subheadings like 'The following information are in relation to Account No' "
            "or 'Aging Information', return true for 'has_trade_reference' and count the number of distinct trade reference entries "
            "in the summary table as 'trade_reference_count'. "
        )

        if self.report_type == ReportType.INDIVIDUAL:
            return (
                common_instructions +
                "From the table 'Credit Info at a Glance', extract: "
                "- 'legal_personal': the number of legal records in past 24 months (personal capacity), from the 'Value' column. "
                "- 'legal_non_personal': the number of legal records in past 24 months (non-personal capacity), from the 'Value' column. "
                "- 'special_attention_accounts_0': the value for 'Special Attention Accounts', from the 'Value' column. "
                "From the section 'C1: BANKING PAYMENT RECORDS (SOURCE: CCRIS, BANK NEGARA MALAYSIA)', "
                "under 'Summary of Potential & Current Liabilities', for the row labeled 'As Borrower': "
                "- 'total_outstanding_balance': the value under the 'Outstanding' column. "
                "- 'total_limit': the value under the 'Total Limit' column. "
                "- 'special_attention_accounts_1': the value ('Y' or 'N') for 'Special Attention Account' from the CCRIS summary. "
                + legal_instructions +
                trade_ref_instructions +
                "Return the extracted data in the following JSON format: "
                "{\"total_outstanding_balance\": value or null, "
                "\"total_limit\": value or null, "
                "\"special_attention_accounts_0\": value or null, "
                "\"special_attention_accounts_1\": value or null, "
                "\"legal_non_personal\": value or null, "
                "\"legal_personal\": value or null, "
                "\"legal_cases_count\": value or null, "
                "\"has_trade_reference\": true or false, "
                "\"trade_reference_count\": value or null, "
            )
        elif self.report_type == ReportType.COMPANY:
            shareholders_fields = ""
            financial_fields = ""
            directors_fields = ""
            partners_fields = ""
            if self.relevant_values.get('partnership') is None:
                shareholders_fields = (
                    "From the table 'Financials and Shareholders', extract: "
                    "- 'paid_up_capital': the value for 'Paid-Up Capital (RM)'. "
                )
                financial_fields = (
                    "From the 'Financial Highlights' / financial statements tables, extract for the latest financial year (second column): "
                    "- 'financial_year_end': the financial year end date in YYYY-MM-DD format. "
                    "- 'current_assets': the current assets value. "
                    "- 'current_liabilities': the current liabilities value. "
                    "- 'non_current_assets': the non-current assets value. "
                    "- 'total_assets': the total assets value. "
                    "- 'non_current_liabilities': the non-current liabilities value. "
                    "- 'long_term_liabilities': the long-term liabilities value. "
                    "- 'total_liabilities': the total liabilities value. "
                    "- 'retained_earning': the retained earning value. "
                    "- 'net_worth': the net worth (TA - TL) value. "
                    "- 'revenue': the revenue value. "
                    "- 'profit_after_tax': the profit / (loss) after tax value. "
                    "- 'current_ratio': the current ratio value. "
                    "- 'gearing_ratio': the gearing ratio value. "
                    "- 'debt_to_equity_ratio': the debt to equity ratio value. "
                )
                directors_fields = (
                    "From the table 'DIRECTORS / OFFICERS', extract 'director_count': the number of directors indicated by the designation 'DS'. Note that if the designation is 'SC', this indicates a company secretary and should not be counted towards the director count. "
                )
            else:
                partners_fields = (
                    "From the table 'B1: BUSINESS PROFILE' and 'CURRENT BUSINESS OWNER(S)/PARTNER(S)', extract 'partner_count': the number of partners indicated by the position 'Partner'."
                )

            return (
                common_instructions +
                "From the table 'A: SNAPSHOT', extract: "
                "- 'registration_date': the registration date. "
                "- 'type': the value for 'Type' or 'Type of Company'. "
                "- 'msic': the MSIC value. "
                "- 'is_partnership': true if you see fields like 'Business Commenced', 'Last Changed Date', "
                "'ROB Search Date', or 'Current Registration Expiry Date' in the Snapshot table, otherwise false. "
                + shareholders_fields +
                "From the table 'Credit Info at a Glance', extract the Entity column values: "
                 "- 'legal_personal_entity': the entity's number of legal records in past 24 months (personal capacity). "
                "- 'legal_non_personal_entity': the entity's number of legal records in past 24 months (non-personal capacity). "
                "- 'special_attention_accounts_entity': the entity's value for 'Special Attention Accounts'. "
                + partners_fields 
                + directors_fields
                + financial_fields +
                "From the section 'C1: BANKING PAYMENT RECORDS (SOURCE: CCRIS, BANK NEGARA MALAYSIA)', "
                "under 'Summary of Potential & Current Liabilities', for the row labeled 'As Borrower': "
                "- 'total_outstanding_balance': the value under the 'Outstanding' column. "
                "- 'total_limit': the value under the 'Total Limit' column. "
                "If section C1: BANKING PAYMENT RECORDS is entirely absent or show 'No Information Available', return null for those fields. "
                + legal_instructions +
                trade_ref_instructions +
                "Return the extracted data in the following JSON format: "
                "{\"registration_date\": value or null, "
                "\"type\": value or null, "
                "\"msic\": value or null, "
                "\"is_partnership\": true or false, "
                "\"total_outstanding_balance\": value or null, "
                "\"total_limit\": value or null, "
                "\"special_attention_accounts_entity\": value or null, "
                "\"legal_non_personal_entity\": value or null, "
                "\"legal_personal_entity\": value or null, "
                "\"legal_cases_count\": value or null, "
                "\"has_trade_reference\": true or false, "
                "\"trade_reference_count\": value or null, "
                + (", \"paid_up_capital\": value or null"
                ", \"financial_year_end\": value or null"
                ", \"revenue\": value or null"
                ", \"profit_after_tax\": value or null"
                ", \"current_assets\": value or null"
                ", \"current_liabilities\": value or null"
                ", \"non_current_assets\": value or null"
                ", \"total_assets\": value or null"
                ", \"non_current_liabilities\": value or null"
                ", \"long_term_liabilities\": value or null"
                ", \"total_liabilities\": value or null"
                ", \"retained_earning\": value or null"
                ", \"net_worth\": value or null"
                ", \"current_ratio\": value or null"
                ", \"gearing_ratio\": value or null"
                ", \"debt_to_equity_ratio\": value or null"
                ", \"director_count\": value or null"
                if self.relevant_values.get('partnership') is None else ", \"partner_count\": value or null") +
                "}."
            )

    def __compare_markdown_results_with_di__(self, markdown_result: Dict, parsed_data: Dict,
                                          extracted_data: Dict, confidence_di: Dict,
                                          confidence_l1: Dict) -> set:
        """Compare markdown results with results from document intelligence layer 0.
        
        For each field:
        - If DI and markdown agree, and both have confidence >= threshold, and validation passes: keep DI value.
        - If they disagree, or either has low confidence, or validation fails: mark field for Layer 2 fallback.
        - If DI is missing but markdown has a value with high confidence: adopt the markdown value.
        
        Returns the set of fields that still need Layer 2 fallback.
        """
        fields_needing_fallback = set()
        required_fields = self.__get_required_fields__()

        # Build a mapping from parsed_data field names to the raw extracted_data / markdown_result keys
        # so we can compare like-for-like values
        if self.report_type == ReportType.INDIVIDUAL:
            field_to_md_keys = {
                'utilisation': ['total_outstanding_balance', 'total_limit'],
                'special_attention_accounts': ['special_attention_accounts_0', 'special_attention_accounts_1'],
                'legal_cases': ['legal_non_personal', 'legal_personal', 'legal_cases_count'],
                'blacklist': ['has_trade_reference', 'trade_reference_count'],
            }
        else:
            field_to_md_keys = {
                'utilisation': ['total_outstanding_balance', 'total_limit'],
                'special_attention_accounts': ['special_attention_accounts_entity'],
                'legal_cases': ['legal_non_personal_entity', 'legal_personal_entity', 'legal_cases_count'],
                'blacklist': ['has_trade_reference', 'trade_reference_count'],
                'years_in_business': ['registration_date'],
                'type_of_company': ['type'],
                'nature_of_business': ['msic'],
                'number_of_directors_or_partners': ['partner_count', 'director_count'],
                'paid_up_capital': ['paid_up_capital'],
                'financial_report_date': ['financial_year_end'],
                'turnover': ['revenue'],
                'net_profit': ['profit_after_tax'],
                'retained_profit': ['retained_earning'], 
                'net_worth': ['net_worth'], 
                'net_current_assets': ['current_assets', 'current_liabilities'],
                'current_ratio': ['current_ratio'],
                'gearing_ratio': ['gearing_ratio', 'debt_to_equity_ratio'],
            }

        for field in required_fields:
            di_value = parsed_data.get(field)

            # Skip fields already marked as N/A (intentionally absent)
            if di_value == 'N/A':
                continue

            # Check DI confidence for this field
            di_conf = self.__get_field_confidence__(field, confidence_di)
            di_high_conf = di_conf is None or di_conf >= LOW_CONFIDENCE_THRESHOLD

            # Check markdown confidence for the corresponding keys
            md_keys = field_to_md_keys.get(field, [])
            md_values_available = md_keys and all(
                markdown_result.get(k) is not None for k in md_keys
            ) if markdown_result else False

            md_high_conf = True
            if md_values_available and confidence_l1:
                for k in md_keys:
                    k_conf = self.__get_field_confidence__(k, confidence_l1)
                    if k_conf is not None and k_conf < LOW_CONFIDENCE_THRESHOLD:
                        md_high_conf = False
                        break

            # --- Compare values ---
            if di_value is not None and md_values_available:
                # Attempt to compare the specific sub-fields
                values_agree = self.__compare_field_values__(field, di_value, markdown_result, extracted_data)

                if values_agree and di_high_conf and md_high_conf:
                    # Both agree with high confidence — keep DI value
                    logger.info("Layer 1: Field '%s' — DI and markdown agree with high confidence, keeping DI value", field)
                    continue
                else:
                    logger.warning("Layer 1: Field '%s' — DI/markdown disagree or low confidence, marking for Layer 2", field)
                    fields_needing_fallback.add(field)

            elif di_value is None and md_values_available and md_high_conf:
                # DI missing but markdown has a high-confidence value — adopt markdown value
                logger.info("Layer 1: Field '%s' — DI missing, adopting markdown value", field)
                self.__adopt_markdown_value__(field, markdown_result, parsed_data, extracted_data)

            elif di_value is None:
                # Both missing or markdown not available
                logger.warning("Layer 1: Field '%s' — missing from both DI and markdown, marking for Layer 2", field)
                fields_needing_fallback.add(field)

            else:
                # DI has value but markdown doesn't have the relevant keys — trust DI if high confidence
                if not di_high_conf:
                    logger.warning("Layer 1: Field '%s' — DI low confidence, no markdown corroboration, marking for Layer 2", field)
                    fields_needing_fallback.add(field)
                else:
                    logger.info("Layer 1: Field '%s' — DI high confidence, no markdown data, keeping DI value", field)

        # Handle partnership detection from markdown
        if markdown_result and markdown_result.get('is_partnership') == True:
            if self.relevant_values.get('partnership') is None:
                self.relevant_values['partnership'] = True
                logger.info("Layer 1: Detected partnership from markdown")

        return fields_needing_fallback


    def __compare_field_values__(self, field: str, di_value, markdown_result: Dict,
                                extracted_data: Dict) -> bool:
        """Compare a parsed DI field value against the corresponding markdown extraction.
        Returns True if the values effectively agree."""
        try:
            if field == 'utilisation':
                md_bal = markdown_result.get('total_outstanding_balance')
                md_limit = markdown_result.get('total_limit')
                if md_bal is not None and md_limit is not None:
                    md_bal_norm = self.__normalize_numeric_str__(str(md_bal))
                    md_limit_norm = self.__normalize_numeric_str__(str(md_limit))
                    # Compare against raw extracted values from DI
                    di_bal = extracted_data.get('total_outstanding_balance_0') or extracted_data.get('total_outstanding_balance_1')
                    di_limit = extracted_data.get('total_limit_0') or extracted_data.get('total_limit_1')
                    if di_bal and di_limit:
                        return (self.__normalize_numeric_str__(di_bal) == md_bal_norm and
                                self.__normalize_numeric_str__(di_limit) == md_limit_norm)
                return False

            elif field == 'special_attention_accounts':
                if self.report_type == ReportType.INDIVIDUAL:
                    md_0 = markdown_result.get('special_attention_accounts_0')
                    md_1 = markdown_result.get('special_attention_accounts_1')
                    if md_0 is not None and md_1 is not None and md_0[0] == md_1[0]:
                        md_val = md_0
                else:
                    md_val = markdown_result.get('special_attention_accounts_entity')
                if md_val is not None and di_value is not None:
                    # Compare first character (e.g. 'N' vs 'NO', 'Y' vs 'YES')
                    return str(di_value)[0].upper() == str(md_val)[0].upper()
                return False

            elif field == 'legal_cases':
                if self.report_type == ReportType.INDIVIDUAL:
                    md_np = markdown_result.get('legal_non_personal')
                    md_p = markdown_result.get('legal_personal')
                else:
                    md_np = markdown_result.get('legal_non_personal_entity')
                    md_p = markdown_result.get('legal_personal_entity')
                if md_np is not None and md_p is not None:
                    md_total = int(md_np) + int(md_p)
                    return int(di_value) == md_total
                return False

            elif field == 'blacklist':
                md_has_tr = markdown_result.get('has_trade_reference')
                md_tr_count = markdown_result.get('trade_reference_count')
                if md_has_tr is False and di_value == 0:
                    return True
                if md_has_tr is True and md_tr_count is not None:
                    return int(di_value) == int(md_tr_count)
                return False

            elif field == 'repayment_to_banks':
                # For repayment_to_banks, compare digit/zero/non-zero tallies between markdown and DI conduct
                md_conduct = markdown_result.get('ccris_conduct')
                di_conduct = extracted_data.get('ccris_conduct')
                if md_conduct is not None and di_conduct is not None and isinstance(di_conduct, list) and len(di_conduct) > 0:
                    # Count digits/zeroes/non-zeroes from markdown conduct (list of lists of ints)
                    if isinstance(md_conduct, list) and len(md_conduct) > 0 and isinstance(md_conduct[0], list):
                        md_digits = sum(len(row) for row in md_conduct)
                        md_zeroes = sum(1 for row in md_conduct for d in row if d == 0)
                        md_non_zeroes = md_digits - md_zeroes
                        
                        # Count digits/zeroes/non-zeroes from DI conduct (list of strings)
                        if isinstance(di_conduct[0], str):
                            di_digits = sum(len(re.sub(r'[^0-9]', '', s)) for s in di_conduct)
                            di_zeroes = sum(s.count('0') for s in di_conduct)
                            di_non_zeroes = di_digits - di_zeroes
                            
                            if md_digits == di_digits and md_zeroes == di_zeroes and md_non_zeroes == di_non_zeroes:
                                logger.info("Cross-validation ccris_conduct tally matches: digits=%d zeroes=%d non_zeroes=%d",
                                           md_digits, md_zeroes, md_non_zeroes)
                                return True
                            else:
                                logger.warning("Cross-validation ccris_conduct tally MISMATCH: md=%d/%d/%d DI=%d/%d/%d",
                                              md_digits, md_zeroes, md_non_zeroes,
                                              di_digits, di_zeroes, di_non_zeroes)
                                return False
                
                # If di_value is 'N/A' and markdown has no CCRIS data
                if di_value == 'N/A' and md_conduct is None:
                    return True
                return di_value is not None  # trust DI if it extracted something

            elif field in ('years_in_business', 'type_of_company', 'nature_of_business'):
                if field == 'years_in_business':
                    md_reg_date = markdown_result.get('registration_date')
                    if md_reg_date is not None:
                        md_years = self.__calculate_years__(md_reg_date)
                        if md_years is not None and di_value is not None:
                            return abs(float(di_value) - float(md_years)) < 0.5
                    return False
                elif field == 'type_of_company':
                    md_type = markdown_result.get('type')
                    if md_type is not None and di_value is not None:
                        # Both should resolve to the same category
                        if self.__is_fuzzy_match__(str(md_type), 'limited by shares private limited'):
                            md_cat = 'Sdn Bhd'
                        else:
                            md_cat = 'Non - Sdn Bhd'
                        return di_value == md_cat
                    return False
                elif field == 'nature_of_business':
                    md_msic = markdown_result.get('msic')
                    if md_msic is not None and di_value is not None:
                        return self.__is_fuzzy_match__(str(di_value), str(md_msic))
                    return False

            elif field in ('turnover', 'net_profit', 'paid_up_capital', 'current_ratio', 'gearing_ratio', 'net_worth', 'retained_profit'):
                key_map = {
                    'turnover': 'revenue',
                    'net_profit': 'profit_after_tax',
                    'paid_up_capital': 'paid_up_capital',
                    'current_ratio': 'current_ratio',
                    'gearing_ratio': 'gearing_ratio',
                    'net_worth': 'net_worth',
                    'retained_profit': 'retained_earning',
                }
                md_key = key_map.get(field)
                md_val = markdown_result.get(md_key)
                if md_val is not None and di_value is not None:
                    try:
                        md_dec = self.__to_decimal__(str(md_val))
                        di_dec = Decimal(str(di_value)) if not isinstance(di_value, Decimal) else di_value
                        return abs(md_dec - di_dec) < Decimal('0.02')
                    except (InvalidOperation, ValueError):
                        return False
                return False

            elif field == 'financial_report_date':
                md_fye = markdown_result.get('financial_year_end')
                if md_fye is not None and di_value is not None:
                    return str(di_value) == str(md_fye)
                return False

            elif field == 'net_current_assets':
                md_ca = markdown_result.get('current_assets')
                md_cl = markdown_result.get('current_liabilities')
                if md_ca is not None and md_cl is not None and di_value is not None:
                    md_nca = self.__to_decimal__(str(md_ca)) - self.__to_decimal__(str(md_cl))
                    di_dec = Decimal(str(di_value)) if not isinstance(di_value, Decimal) else di_value
                    return abs(md_nca - di_dec) < Decimal('0.02')
                return False

            else:
                # For any unmapped fields, we can't compare — treat as agreeing if DI has a value
                return di_value is not None

        except (ValueError, TypeError, InvalidOperation) as e:
            logger.warning("Comparison error for field '%s': %s", field, e)
            return False


    def __adopt_markdown_value__(self, field: str, markdown_result: Dict,
                                parsed_data: Dict, extracted_data: Dict):
        """Adopt a value from the markdown extraction into parsed_data when DI is missing."""
        try:
            if field == 'utilisation':
                md_bal = markdown_result.get('total_outstanding_balance')
                md_limit = markdown_result.get('total_limit')
                if md_bal is not None and md_limit is not None:
                    bal = self.__to_decimal__(str(md_bal))
                    limit = self.__to_decimal__(str(md_limit))
                    if limit > 0:
                        parsed_data['utilisation'] = bal / limit * 100

            elif field == 'special_attention_accounts':
                if self.report_type == ReportType.INDIVIDUAL:
                    md_0 = markdown_result.get('special_attention_accounts_0')
                    md_1 = markdown_result.get('special_attention_accounts_1')
                    if md_0 is not None and md_1 is not None and md_0[0] == md_1[0]:
                        md_val = md_0
                else:
                    md_val = markdown_result.get('special_attention_accounts_entity')
                if md_val is not None:
                    parsed_data['special_attention_accounts'] = str(md_val)

            elif field == 'legal_cases':
                if self.report_type == ReportType.INDIVIDUAL:
                    md_np = markdown_result.get('legal_non_personal')
                    md_p = markdown_result.get('legal_personal')
                else:
                    md_np = markdown_result.get('legal_non_personal_entity')
                    md_p = markdown_result.get('legal_personal_entity')
                if md_np is not None and md_p is not None:
                    parsed_data['legal_cases'] = int(md_np) + int(md_p)

            elif field == 'blacklist':
                md_has_tr = markdown_result.get('has_trade_reference')
                md_tr_count = markdown_result.get('trade_reference_count')
                if md_has_tr is False:
                    parsed_data['blacklist'] = 0
                elif md_has_tr is True and md_tr_count is not None:
                    parsed_data['blacklist'] = int(md_tr_count)

            elif field == 'years_in_business':
                md_reg_date = markdown_result.get('registration_date')
                if md_reg_date is not None:
                    years = self.__calculate_years__(md_reg_date)
                    if years is not None:
                        parsed_data['years_in_business'] = years

            elif field == 'type_of_company':
                md_type = markdown_result.get('type')
                if md_type is not None:
                    if self.__is_fuzzy_match__(str(md_type), 'limited by shares private limited'):
                        parsed_data['type_of_company'] = 'Sdn Bhd'
                    else:
                        parsed_data['type_of_company'] = 'Non - Sdn Bhd'

            elif field == 'nature_of_business':
                md_msic = markdown_result.get('msic')
                if md_msic is not None:
                    parsed_data['nature_of_business'] = str(md_msic)

            elif field == 'paid_up_capital':
                md_val = markdown_result.get('paid_up_capital')
                if md_val is not None:
                    parsed_data['paid_up_capital'] = self.__to_decimal__(str(md_val))

            elif field == 'financial_report_date':
                md_fye = markdown_result.get('financial_year_end')
                if md_fye is not None:
                    parsed_data['financial_report_date'] = str(md_fye)

            elif field == 'turnover':
                md_val = markdown_result.get('revenue')
                if md_val is not None:
                    parsed_data['turnover'] = self.__to_decimal__(str(md_val))

            elif field == 'net_profit':
                md_val = markdown_result.get('profit_after_tax')
                if md_val is not None:
                    parsed_data['net_profit'] = self.__to_decimal__(str(md_val))

            elif field == 'net_current_assets':
                md_ca = markdown_result.get('current_assets')
                md_cl = markdown_result.get('current_liabilities')
                if md_ca is not None and md_cl is not None:
                    ca = self.__to_decimal__(str(md_ca))
                    cl = self.__to_decimal__(str(md_cl))
                    parsed_data['net_current_assets'] = ca - cl
            
            elif field == 'net_worth':
                md_val = markdown_result.get('net_worth')
                if md_val is not None:
                    parsed_data['net_worth'] = self.__to_decimal__(str(md_val))
            
            elif field == 'retained_profit':
                md_val = markdown_result.get('retained_earning')
                if md_val is not None:
                    parsed_data['retained_profit'] = self.__to_decimal__(str(md_val))

            elif field == 'current_ratio':
                md_val = markdown_result.get('current_ratio')
                if md_val is not None:
                    parsed_data['current_ratio'] = self.__to_decimal__(str(md_val))

            elif field == 'gearing_ratio':
                md_val = markdown_result.get('gearing_ratio')
                if md_val is not None:
                    parsed_data['gearing_ratio'] = self.__to_decimal__(str(md_val))

        except (ValueError, TypeError, InvalidOperation) as e:
            logger.warning("Failed to adopt markdown value for field '%s': %s", field, e)
    
    def __run_layer2_image_fallback__(self, extracted_data: Dict, parsed_data: Dict,
                                     fields_needing_fallback: set, markdown_result: Dict):
        """Runs targeted image extraction for fields still needing fallback."""

        # Map parsed_data fields to the table tags used for image extraction
        field_to_tags = {
            'utilisation': ['ccris_summary'],
            'repayment_to_banks': ['ccris_detail'],
            'special_attention_accounts': ['credit_info_at_a_glance', 'ccris_summary'],
            'legal_cases': ['credit_info_at_a_glance'],
            'years_in_business': ['snapshot'],
            'type_of_company': ['snapshot'],
            'nature_of_business': ['snapshot'],
            'paid_up_capital': ['financials_and_shareholders'],
            'financial_report_date': ['financial_statements'],
            'turnover': ['financial_statements'],
            'net_profit': ['financial_statements'],
            'retained_profit': ['financial_statements'],
            'net_worth': ['financial_statements'],
            'net_current_assets': ['financial_statements'],
            'current_ratio': ['financial_statements'],
            'gearing_ratio': ['financial_statements'],
            'blacklist': ['trade_reference'],
            'number_of_directors_or_partners': ['directors_officers', 'business_profile'],
        }

        # Collect unique tags to avoid duplicate API calls
        tags_to_call = set()
        for field in fields_needing_fallback:
            for tag in field_to_tags.get(field, []):
                tags_to_call.add(tag)

        # Special handling: if repayment_to_banks needs fallback and we have conduct but no balance/limit
        if 'repayment_to_banks' in fields_needing_fallback:
            if extracted_data.get('ccris_conduct') is not None and (
                    extracted_data.get('total_outstanding_balance_1') is None or
                    extracted_data.get('total_limit_1') is None):
                tags_to_call.discard('ccris_detail')
                tags_to_call.add('ccris_detail_edge_case')

        logger.info("Layer 2 image tags to call: %s", tags_to_call)

        # Execute each image extraction and merge results
        for tag in tags_to_call:
            try:
                image_result, image_choice = self.extract_using_markdown_and_image_with_confidence(tag)
            except Exception as e:
                logger.error("Layer 2 image extraction failed for tag '%s': %s", tag, e, exc_info=True)
                continue

            if not image_result:
                logger.warning("Layer 2 image extraction returned no result for tag '%s'", tag)
                continue

            # Evaluate OpenAI confidence for this extraction
            confidence_l2 = {}
            if image_choice:
                confidence_l2 = evaluate_confidence_openai(
                    extract_result=image_result,
                    choice=image_choice
                )
                logger.info("Layer 2 confidence for tag '%s': %s", tag, confidence_l2)

            # Merge based on tag
            if tag == 'ccris_summary':
                if parsed_data.get('utilisation') is None:
                    if image_result.get('total_outstanding_balance') is not None and image_result.get('total_limit') is not None:
                        bal = self.__to_decimal__(str(image_result['total_outstanding_balance']))
                        limit = self.__to_decimal__(str(image_result['total_limit']))
                        if limit > 0:
                            parsed_data['utilisation'] = bal / limit * 100
                if parsed_data.get('special_attention_accounts') is None and image_result.get('special_attention_accounts') is not None:
                    parsed_data['special_attention_accounts'] = image_result['special_attention_accounts']

            elif tag in ('ccris_detail', 'ccris_detail_edge_case'):
                if image_result.get('ccris_conduct') is not None:
                    conduct = image_result['ccris_conduct']
                    di_conduct = extracted_data.get('ccris_conduct')
                    if isinstance(conduct, list) and len(conduct) > 0 and di_conduct is not None:
                        # Tally check against md extraction
                        if isinstance(conduct[0], list):
                            total_digits = sum(len(row) for row in conduct)
                            total_zeroes = sum(1 for row in conduct for d in row if d == 0)
                            total_non_zeroes = total_digits - total_zeroes

                            # Count digits/zeroes/non-zeroes from DI conduct (list of strings)
                            if isinstance(di_conduct, list) and len(di_conduct) > 0 and isinstance(di_conduct[0], str):
                                di_digits = sum(len(re.sub(r'[^0-9]', '', s)) for s in di_conduct)
                                di_zeroes = sum(s.count('0') for s in di_conduct)
                                di_non_zeroes = di_digits - di_zeroes
                                
                                if total_digits == di_digits and total_zeroes == di_zeroes and total_non_zeroes == di_non_zeroes:
                                    logger.info("Cross-validation ccris_conduct tally matches: digits=%d zeroes=%d non_zeroes=%d",
                                            total_digits, total_zeroes, total_non_zeroes)

                                else:
                                    logger.warning("Cross-validation ccris_conduct tally MISMATCH: md=%d/%d/%d DI=%d/%d/%d",
                                                total_digits, total_zeroes, total_non_zeroes,
                                                di_digits, di_zeroes, di_non_zeroes)
                                    if total_non_zeroes == di_non_zeroes:
                                        logger.warning("Despite tally mismatch, non-zero count matches for ccris_conduct, which may be most indicative of repayment behavior.")
                                    
                            logger.info("Document Intelligence extraction of CCRIS details prioritized due to lower hallucination risk.")

                if extracted_data.get('total_outstanding_balance_1') is None and image_result.get('total_outstanding_balance') is not None:
                    extracted_data['total_outstanding_balance_1'] = str(image_result['total_outstanding_balance'])
                if extracted_data.get('total_limit_1') is None and image_result.get('total_limit') is not None:
                    extracted_data['total_limit_1'] = str(image_result['total_limit'])

                # Re-try utilisation with updated data
                if parsed_data.get('utilisation') is None:
                    util_keys = ['total_outstanding_balance_0', 'total_outstanding_balance_1', 'total_limit_0', 'total_limit_1']
                    if all(extracted_data.get(key) is not None for key in util_keys):
                        bal = self.__to_decimal__(extracted_data['total_outstanding_balance_0'])
                        limit = self.__to_decimal__(extracted_data['total_limit_0'])
                        if limit > 0:
                            parsed_data['utilisation'] = bal / limit * 100

            elif tag == 'credit_info_at_a_glance':
                if parsed_data.get('special_attention_accounts') is None:
                    spa_key = 'special_attention_accounts_entity' if self.report_type == ReportType.COMPANY else 'special_attention_accounts'
                    if image_result.get(spa_key) is not None:
                        parsed_data['special_attention_accounts'] = image_result[spa_key]

                if parsed_data.get('legal_cases') is None:
                    if self.report_type == ReportType.INDIVIDUAL:
                        lnp = image_result.get('legal_non_personal')
                        lp = image_result.get('legal_personal')
                    else:
                        lnp = image_result.get('legal_non_personal_entity')
                        lp = image_result.get('legal_personal_entity')
                    if lnp is not None and lp is not None:
                        try:
                            lnp_int = int(lnp)
                            lp_int = int(lp)
                            parsed_data['legal_cases'] = lnp_int + lp_int
                        except (ValueError, TypeError):
                            if str(lnp) == '0' and str(lp) == '0':
                                parsed_data['legal_cases'] = 0

            elif tag == 'snapshot':
                if parsed_data.get('years_in_business') is None and image_result.get('registration_date') is not None:
                    years = self.__calculate_years__(image_result['registration_date'])
                    if years is not None:
                        parsed_data['years_in_business'] = years
                if parsed_data.get('type_of_company') is None and image_result.get('type') is not None:
                    type_val = image_result['type']
                    if self.__is_fuzzy_match__(type_val, 'limited by shares private limited'):
                        parsed_data['type_of_company'] = 'Sdn Bhd'
                    else:
                        parsed_data['type_of_company'] = 'Non - Sdn Bhd'
                if parsed_data.get('nature_of_business') is None and image_result.get('msic') is not None:
                    parsed_data['nature_of_business'] = image_result['msic']
                if image_result.get('is_partnership') == True:
                    self.relevant_values['partnership'] = True

            elif tag == 'financials_and_shareholders':
                if parsed_data.get('paid_up_capital') is None and image_result.get('paid_up_capital') is not None:
                    parsed_data['paid_up_capital'] = self.__to_decimal__(str(image_result['paid_up_capital']))

            elif tag == 'financial_statements':
                logger.info("Layer 2 financial statements data: %s", json.dumps(image_result, indent=2))
                if parsed_data.get('financial_report_date') is None and image_result.get('financial_year_end') is not None:
                    parsed_data['financial_report_date'] = image_result['financial_year_end']
                if parsed_data.get('turnover') is None and image_result.get('revenue') is not None:
                    parsed_data['turnover'] = self.__to_decimal__(str(image_result['revenue']))
                if parsed_data.get('net_profit') is None and image_result.get('profit_after_tax') is not None:
                    parsed_data['net_profit'] = self.__to_decimal__(str(image_result['profit_after_tax']))
                if parsed_data.get('retained_profit') is None and image_result.get('retained_earning') is not None:
                    parsed_data['retained_profit'] = self.__to_decimal__(str(image_result['retained_earning']))
                if parsed_data.get('net_worth') is None and image_result.get('net_worth') is not None:
                    parsed_data['net_worth'] = self.__to_decimal__(str(image_result['net_worth']))

                if parsed_data.get('net_current_assets') is None:
                    fs_keys = ['current_assets', 'current_liabilities', 'non_current_assets', 'total_assets',
                               'non_current_liabilities', 'long_term_liabilities', 'total_liabilities']
                    if all(image_result.get(k) is not None for k in fs_keys):
                        nca = self.__to_decimal__(str(image_result['non_current_assets']))
                        ca = self.__to_decimal__(str(image_result['current_assets']))
                        ta = self.__to_decimal__(str(image_result['total_assets']))
                        ncl = self.__to_decimal__(str(image_result['non_current_liabilities']))
                        cl = self.__to_decimal__(str(image_result['current_liabilities']))
                        ltl = self.__to_decimal__(str(image_result['long_term_liabilities']))
                        tl = self.__to_decimal__(str(image_result['total_liabilities']))
                        if (ta == nca + ca) and (tl == ncl + cl + ltl):
                            parsed_data['net_current_assets'] = ca - cl
                        else:
                            logger.error("Layer 2: Balance sheet validation failed")

                if parsed_data.get('current_ratio') is None and image_result.get('current_ratio') is not None:
                    extracted_cr = self.__to_decimal__(str(image_result['current_ratio']))
                    if image_result.get('current_assets') is not None and image_result.get('current_liabilities') is not None:
                        ca = self.__to_decimal__(str(image_result['current_assets']))
                        cl = self.__to_decimal__(str(image_result['current_liabilities']))
                        cr = ca / cl if cl > 0 else Decimal(0)
                        if abs(cr - extracted_cr) < Decimal('0.01'):
                            parsed_data['current_ratio'] = extracted_cr

                if parsed_data.get('gearing_ratio') is None and image_result.get('gearing_ratio') is not None:
                    if image_result.get('total_liabilities') is not None and image_result.get('net_worth') is not None:
                        tl = self.__to_decimal__(str(image_result['total_liabilities']))
                        nw = self.__to_decimal__(str(image_result['net_worth']))
                        extracted_gr = self.__to_decimal__(str(image_result['gearing_ratio']))
                        calc_gr = tl / nw if nw > 0 else Decimal(0)
                        if abs(extracted_gr - calc_gr) < Decimal('0.01'):
                            parsed_data['gearing_ratio'] = extracted_gr

    def extract_using_markdown_with_confidence(self, prompt: str) -> Tuple[Optional[Dict], Optional[Any]]:
        """Extract data from markdown and return (result_dict, choice) for confidence evaluation."""
        markdown = self.result.content
        if not markdown:
            return None, None

        client = self.__get_openai_client__(self.options)

        user_content = [{"type": "text", "text": prompt}, {"type": "text", "text": markdown}]

        try:
            completion = client.chat.completions.create(
                model=self.options.deployment_name,
                messages=[
                    {"role": "system", "content": self.options.system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=self.options.max_tokens,
                temperature=self.options.temperature,
                top_p=self.options.top_p,
                logprobs=True,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            logger.error("Markdown extraction failed: %s", e, exc_info=True)
            return None, None

        choice = completion.choices[0]
        raw_content = choice.message.content

        try:
            response_obj_dict = json.loads(raw_content)
            return response_obj_dict, choice
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from markdown extraction: %s", raw_content[:200])
            return None, None

    def extract_using_markdown_and_image_with_confidence(self, table_tag: str) -> Tuple[Optional[Dict], Optional[Any]]:
        """Extract data from markdown + images and return (result_dict, choice) for confidence evaluation."""
        markdown = self.result.content
        if not markdown:
            return None, None

        client = self.__get_openai_client__(self.options)

        page_start, page_end = self.__get_page_range_for_table_tag__(table_tag)
        if page_start is None or page_end is None:
            page_start, page_end = 1, len(self.result.pages)

        image_uris = self.__get_document_image_uris__(self.bytes, page_start, page_end)
        table_prompt = self.__get_prompt_for_table_tag__(table_tag)

        user_content = [{"type": "text", "text": table_prompt}, {"type": "text", "text": markdown}]
        for image_uri in image_uris:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": image_uri, "detail": "high"}
            })

        try:
            completion = client.chat.completions.create(
                model=self.options.deployment_name,
                messages=[
                    {"role": "system", "content": self.options.system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=self.options.max_tokens,
                temperature=self.options.temperature,
                top_p=self.options.top_p,
                logprobs=True,
                response_format={"type": "json_object"}
            )
        except Exception as e:
            logger.error("Image extraction failed for tag '%s': %s", table_tag, e, exc_info=True)
            return None, None

        choice = completion.choices[0]
        raw_content = choice.message.content

        try:
            response_obj_dict = json.loads(raw_content)
            return response_obj_dict, choice
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from image extraction (tag=%s): %s", table_tag, raw_content[:200])
            return None, None

    def extract_using_markdown(self, prompt: str):
        """Extract data from markdown generated from Azure Document Intelligence."""
        result, _ = self.extract_using_markdown_with_confidence(prompt)
        return result

    def extract_using_markdown_and_image(self, table_tag: str):
        """Extract data from markdown content and images of document pages."""
        result, _ = self.extract_using_markdown_and_image_with_confidence(table_tag)
        return result

    def __identify_tables_from_json__(self) -> List[Dict]:
        """Identify relevant tables."""
        
        tagged_tables = []
        
        if not self.result.tables:
            logger.warning("No tables found in document")
            return tagged_tables
        
        logger.info("Processing %d tables for tagging", len(self.result.tables))
        paragraphs = self.result.paragraphs or []

        consumed_indices = set()
        
        for table_idx, table in enumerate(self.result.tables):
            if table_idx in consumed_indices: 
                continue

            table_region = table.bounding_regions[0] if table.bounding_regions else None
            
            # Find preceding paragraph to use as context
            preceding_text = self.__find_missing_header__(table_region, paragraphs)
            
            # Determine table type from headers or context
            idx_and_type = self.__determine_table_type__(table, preceding_text, table_idx)

            if len(idx_and_type) == 1:
                tag_type = idx_and_type[0]['type']
                if tag_type == 'UNKNOWN':
                    continue

                lookup_table = self.__convert_to_row_map__(table)   
                logger.info("Tagged table index %d as type %s", idx_and_type[0]['idx'], tag_type)
                consumed_indices.add(idx_and_type[0]['idx'])
                tagged_tables.append({
                    'type': tag_type,
                    'table': lookup_table,
                    'raw_table': table
                })
                self.__record_table_pages__(idx_and_type[0]['idx'], tag_type)
            elif len(idx_and_type) > 1:
                # multi-page CCRIS Details tables identified
                for item in idx_and_type:
                    lookup_table = self.__convert_to_row_map__(self.result.tables[item['idx']]) 
                    logger.info("Tagged table index %d as type %s", item['idx'], item['type'])
                    tagged_tables.append({
                        'type': item['type'],
                        'table': lookup_table,
                        'raw_table': self.result.tables[item['idx']]
                    })
                    self.__record_table_pages__(item['idx'], item['type'])
                    consumed_indices.add(item['idx'])

        logger.info("Tagged %d tables after processing", len(tagged_tables))
        return tagged_tables

    def __find_paragraph_below_paragraph__(self, paragraph, paragraphs) -> Optional[str]:
        """Finds the paragraph immediately below a given paragraph."""
        if not paragraph or not paragraphs:
            return None
        
        para_page = paragraph.bounding_regions[0].page_number if paragraph.bounding_regions else None
        para_bottom = paragraph.bounding_regions[0].polygon[5] if paragraph.bounding_regions else None  # Y-coordinate of bottom-left
        
        # Find paragraphs on same page that start after this paragraph ends
        candidates = []
        for para in paragraphs:
            if not para.bounding_regions:
                continue
            
            para_region = para.bounding_regions[0]
            if para_region.page_number == para_page:
                para_top = para_region.polygon[1]  # Y-coordinate of top-left
                # If the paragraph is within 0.5 inches below the given paragraph
                if 0 < (para_top - para_bottom) < 0.5: 
                    candidates.append((para_top, para.content))
        
        # Return the closest following heading
        if candidates:
            candidates.sort()  # Closest first
            val = candidates[0][1]
            return val.strip().lower()
        
        return None

    def __find_missing_header__(self, table_region, paragraphs) -> Optional[str]:
        """Finds the paragraph immediately before a table using bounding regions."""
        if not table_region or not paragraphs:
            return None
        
        table_page = table_region.page_number
        table_top = table_region.polygon[1]  # Y-coordinate of top-left
        
        # Find paragraphs on same page that end before table starts
        candidates = []
        for para in paragraphs:
            if not para.bounding_regions:
                continue
            
            para_region = para.bounding_regions[0]
            if para_region.page_number == table_page:
                para_bottom = para_region.polygon[5]  # Y-coordinate of bottom-left
                # If the paragraph is within 0.5 inches above the table
                if 0 < (table_top - para_bottom) < 0.5: 
                    candidates.append((para_bottom, para.content))
        
        # Return the closest preceding heading
        if candidates:
            candidates.sort(reverse=True)  # Closest first
            return candidates[0][1]
        
        return None

    def __is_fuzzy_match__(self, a: str, b: str, threshold: int = 90) -> bool:
        """Checks if two strings are a fuzzy match above the given threshold."""
        return fuzz.partial_ratio(a.strip().lower(), b.strip().lower()) > threshold
    
    def __determine_table_type__(self, table, preceding_text: Optional[str], table_idx: int) -> List[Dict]:
        """Determines table type from headers or preceding context."""

        # First try: Extract header cells
        headers = []
        for cell in table.cells:
            # since headers may not be considered 'columnHeader' by SDK, we check first row
            if cell.row_index == 0:
                headers.append(cell.content.strip().lower())
        header_text = ' '.join(headers)

        if (self.__is_fuzzy_match__(header_text, 'c1: banking payment records (source: ccris, bank negara malaysia)')
        or self.__is_fuzzy_match__(header_text, 'ccris entity key')
        or self.__is_fuzzy_match__(header_text, 'ccris summary')
        or self.__is_fuzzy_match__(header_text, 'credit applications')
        or self.__is_fuzzy_match__(header_text, 'approved in past 12 months')
        or self.__is_fuzzy_match__(header_text, 'summary of potential & current liabilities')
        or self.__is_fuzzy_match__(header_text, 'as borrower')):
            return [{'idx': table_idx, 'type': 'CCRIS_SUMMARY'}]
            
        if self.__is_fuzzy_match__(header_text, 'ccris details)') or self.__is_fuzzy_match__(header_text, 'loan information') or self.__is_fuzzy_match__(header_text, 'outstanding credit') or (self.__is_fuzzy_match__(header_text, 'no') and (table.column_count == 25 or table.column_count == 14)):
            return self.__handle_ccris_details_tables__(table_idx)
        
        if self.__is_fuzzy_match__(header_text, 'd1: legal cases (subject as defendant)') or self.__is_fuzzy_match__(header_text, 'd2: legal cases (subject as plaintiff)'):
            if table.column_count == 6: # summary table
                return [{'idx': table_idx, 'type': 'LEGAL_CASES_SUMMARY'}]
            return [{'idx': table_idx, 'type': 'LEGAL_CASES'}]
        
        if self.__is_fuzzy_match__(header_text, 'e2: trade reference') or self.__is_fuzzy_match__(header_text, 'the following information are in relation to account no:') or self.__is_fuzzy_match__(header_text, '1. relationship') or self.__is_fuzzy_match__(header_text, '2. aging information'):
            if table.column_count == 6: # summary table
                return [{'idx': table_idx, 'type': 'TRADE_REFERENCE_SUMMARY'}]
            return [{'idx': table_idx, 'type': 'TRADE_REFERENCE'}]
            
        if self.report_type == ReportType.INDIVIDUAL:

            if self.__is_fuzzy_match__(header_text, 'credit info at a glance') or self.__is_fuzzy_match__(header_text, 'credit info') or self.__is_fuzzy_match__(header_text, 'bankruptcy proceedings record'):
                return [{'idx': table_idx, 'type': 'CREDIT_INFO_AT_A_GLANCE'}]
                        
        elif self.report_type == ReportType.COMPANY:

            if self.__is_fuzzy_match__(header_text, 'a: snapshot') or self.__is_fuzzy_match__(header_text, 'id verification') or self.__is_fuzzy_match__(header_text, 'company name (your input)') or self.__is_fuzzy_match__(header_text, 'business name (your input)'):
                return [{'idx': table_idx, 'type': 'SNAPSHOT'}]
            
            if (self.__is_fuzzy_match__(header_text, 'financials and shareholders') or self.__is_fuzzy_match__(header_text, 'last updated')) and table.column_count == 2:
                return [{'idx': table_idx, 'type': 'FINANCIALS_AND_SHAREHOLDERS'}]
            
            if self.__is_fuzzy_match__(header_text, 'credit info at a glance') or self.__is_fuzzy_match__(header_text, 'credit info') or self.__is_fuzzy_match__(header_text, 'winding up / bankruptcy proceedings record'):
                return [{'idx': table_idx, 'type': 'CREDIT_INFO_AT_A_GLANCE'}]
            
            # company: sdn bhd
            if self.__is_fuzzy_match__(header_text, 'directors / officers') or (self.__is_fuzzy_match__(header_text, 'name') and table.column_count == 7):
                return [{'idx': table_idx, 'type': 'DIRECTORS_OFFICERS'}]
            
            # company: partnership
            if self.__is_fuzzy_match__(header_text, 'b1: business profile') or self.__is_fuzzy_match__(header_text, 'current business owner(s) / partner(s)') or self.__is_fuzzy_match__(header_text, 'note: the information above have been extracted from ROB computer printout search. We do not warrant as to its accuracy, correctness or completeness. If there are inconsistencies, inaccuracies or missing details or information, please conduct a further probe.'):
                return [{'idx': table_idx, 'type': 'BUSINESS_PROFILE'}]
            
            if (self.__is_fuzzy_match__(header_text, 'financial highlights') or self.__is_fuzzy_match__(header_text, 'financial year end') or self.__is_fuzzy_match__(header_text, 'date of tabling') or self.__is_fuzzy_match__(header_text, 'balance sheet') or self.__is_fuzzy_match__(header_text, 'non-current assets') or self.__is_fuzzy_match__(header_text, 'income statement') or self.__is_fuzzy_match__(header_text, 'revenue') or self.__is_fuzzy_match__(header_text, 'liquidity ratios') or self.__is_fuzzy_match__(header_text, 'current ratio')) and table.column_count == 6:
                self.relevant_values['financial_statements'] = True
                return [{'idx': table_idx, 'type': 'FINANCIAL_STATEMENTS'}]

        # Second try: Use preceding paragraph
        if preceding_text:
            preceding_lower = preceding_text.strip().lower()

            if self.__is_fuzzy_match__(preceding_lower, 'c1: banking payment records (source: ccris, bank negara malaysia)') or self.__is_fuzzy_match__(preceding_lower, 'ccris entity key') or self.__is_fuzzy_match__(preceding_lower, 'ccris summary') or self.__is_fuzzy_match__(preceding_lower, 'credit applications') or self.__is_fuzzy_match__(preceding_lower, 'approved in past 12 months') or self.__is_fuzzy_match__(preceding_lower, 'summary of potential & current liabilities') or self.__is_fuzzy_match__(preceding_lower, 'as borrower'):
                return [{'idx': table_idx, 'type': 'CCRIS_SUMMARY'}]
                
            if self.__is_fuzzy_match__(preceding_lower, 'ccris details)') or self.__is_fuzzy_match__(preceding_lower, 'loan information') or self.__is_fuzzy_match__(preceding_lower, 'outstanding credit') or (self.__is_fuzzy_match__(preceding_lower, 'no') and (table.column_count == 25 or table.column_count == 14)):
                return self.__handle_ccris_details_tables__(table_idx)
            
            if self.__is_fuzzy_match__(preceding_lower, 'd1: legal cases (subject as defendant)') or self.__is_fuzzy_match__(preceding_lower, 'd2: legal cases (subject as plaintiff)'):
                return [{'idx': table_idx, 'type': 'LEGAL_CASES'}]
            
            if self.__is_fuzzy_match__(preceding_lower, 'e2: trade reference') or self.__is_fuzzy_match__(preceding_lower, 'the following information are in relation to account no:') or self.__is_fuzzy_match__(preceding_lower, '1. relationship') or self.__is_fuzzy_match__(preceding_lower, '2. aging information'):
                return [{'idx': table_idx, 'type': 'TRADE_REFERENCE'}]
        
            if self.report_type == ReportType.INDIVIDUAL:  

                if self.__is_fuzzy_match__(preceding_lower, 'credit info at a glance') or self.__is_fuzzy_match__(preceding_lower, 'credit info') or self.__is_fuzzy_match__(preceding_lower, 'bankruptcy proceedings record'):
                    return [{'idx': table_idx, 'type': 'CREDIT_INFO_AT_A_GLANCE'}]
                
            elif self.report_type == ReportType.COMPANY:

                if self.__is_fuzzy_match__(preceding_lower, 'a: snapshot') or self.__is_fuzzy_match__(preceding_lower, 'id verification') or self.__is_fuzzy_match__(preceding_lower, 'company name (your input)'):
                    return [{'idx': table_idx, 'type': 'SNAPSHOT'}]
                
                if (self.__is_fuzzy_match__(preceding_lower, 'financials and shareholders') or self.__is_fuzzy_match__(preceding_lower, 'last updated')) and table.column_count == 2:
                    return [{'idx': table_idx, 'type': 'FINANCIALS_AND_SHAREHOLDERS'}]
                
                if self.__is_fuzzy_match__(preceding_lower, 'credit info at a glance') or self.__is_fuzzy_match__(preceding_lower, 'credit info') or self.__is_fuzzy_match__(preceding_lower, 'winding up / bankruptcy proceedings record'):
                    return [{'idx': table_idx, 'type': 'CREDIT_INFO_AT_A_GLANCE'}]
                
                # company: sdn bhd
                if self.__is_fuzzy_match__(preceding_lower, 'directors / officers') or (self.__is_fuzzy_match__(preceding_lower, 'name') and table.column_count == 7):
                    return [{'idx': table_idx, 'type': 'DIRECTORS_OFFICERS'}]
                
                # company: partnership
                if self.__is_fuzzy_match__(preceding_lower, 'b1: business profile') or self.__is_fuzzy_match__(preceding_lower, 'current business owner(s) / partner(s)') or self.__is_fuzzy_match__(preceding_lower, 'note: the information above have been extracted from ROB computer printout search. We do not warrant as to its accuracy, correctness or completeness. If there are inconsistencies, inaccuracies or missing details or information, please conduct a further probe.'):
                    return [{'idx': table_idx, 'type': 'BUSINESS_PROFILE'}]
                
                if (self.__is_fuzzy_match__(preceding_lower, 'financial highlights') or self.__is_fuzzy_match__(preceding_lower, 'financial year end') or self.__is_fuzzy_match__(preceding_lower, 'date of tabling') or self.__is_fuzzy_match__(preceding_lower, 'balance sheet') or self.__is_fuzzy_match__(preceding_lower, 'non-current assets') or self.__is_fuzzy_match__(preceding_lower, 'income statement') or self.__is_fuzzy_match__(preceding_lower, 'revenue') or self.__is_fuzzy_match__(preceding_lower, 'liquidity ratios') or self.__is_fuzzy_match__(preceding_lower, 'current ratio')) and table.column_count == 6:
                    self.relevant_values['financial_statements'] = True
                    return [{'idx': table_idx, 'type': 'FINANCIAL_STATEMENTS'}]
                
        return [{'idx': table_idx, 'type': 'UNKNOWN'}]

    def __classify_report_type__(self) -> ReportType:
        """Classifies the report type as individual or company based on presence of snapshot table."""
        for table in self.result.tables:
            keywords = []
            for cell in table.cells:
                if cell.column_index == 0:
                    keywords.append(cell.content.strip().lower())
            kw_text = ' '.join(keywords)
            if self.__is_fuzzy_match__(kw_text, 'date of birth') or self.__is_fuzzy_match__(kw_text, 'nationality'):
                return ReportType.INDIVIDUAL
        return ReportType.COMPANY

    def __convert_to_row_map__(self, table):
        """Converts flat cells to nested dictionary with row_index as key and column_index as sub-key."""
        row_map = {}
        for cell in table.cells:
            row_idx = cell.row_index
            col_idx = cell.column_index
            if row_idx not in row_map:
                row_map[row_idx] = {}
            row_map[row_idx][col_idx] = cell.content
        return row_map

    def __handle_ccris_details_tables__(self, table_idx: int) -> List[Dict]:
        """Handles multi-page CCRIS Details tables by tagging them appropriately."""
        next_tables = self.__detect_ccris_details_tables__(table_idx)
        table_types = []
        if len(next_tables) == 0:
            table_types.append({'idx': table_idx, 'type': 'CCRIS_DETAILS_SINGLE'})
        elif len(next_tables) == 1:
            table_types.append({'idx': table_idx, 'type': 'CCRIS_DETAILS_MULTI_START'})
            table_types.append({'idx': next_tables[0], 'type': 'CCRIS_DETAILS_MULTI_END'})
        else:
            table_types.append({'idx': table_idx, 'type': 'CCRIS_DETAILS_MULTI_START'})
            table_types.append({'idx': next_tables[-1], 'type': 'CCRIS_DETAILS_MULTI_END'})
            for idx in range(1, len(next_tables) - 1):
                table_types.append({'idx': next_tables[idx], 'type': 'CCRIS_DETAILS_MULTI_MID'})
        return table_types

    def __detect_ccris_details_tables__(self, table_idx: int) -> List[int]:
        """Detect multi-page CCRIS Details tables."""
        next_tables: List[int] = []
        for idx in range(table_idx, len(self.result.tables) - 1):
            current_table = self.result.tables[idx]
            next_table = self.result.tables[idx + 1]

            # Since CCRIS Details tables have a distinct column count compared to all other tables.
            # We do not check for the boilerplate text between tables since regex approach breaks from OCR errors and fuzzy matching of long boilerplate is slow.
            if current_table.column_count == next_table.column_count:
                next_tables.append(idx + 1)
            else:
                break
        return next_tables                

    def __extract_from_tagged_tables__(self, tagged_tables: List[Dict]) -> Dict[str, List[Dict]]:
        """Extracts data from tagged tables"""
        extracted = {}
        for tagged_table in tagged_tables:
            gen = self.__extract_from_table_general__(tagged_table)
            extracted.update(gen)

            if (self.report_type == ReportType.INDIVIDUAL):
                values = self.__extract_from_table_individual__(tagged_table)
            elif (self.report_type == ReportType.COMPANY):
                values = self.__extract_from_table_company__(tagged_table)
            
            extracted.update(values)
        return extracted

    def __extract_from_table_general__(self, tagged_table: Dict) -> Dict:
        """Extracts data from a single tagged table regardless of report type."""
        table_type = tagged_table['type']
        table = tagged_table['table']
        raw_table = tagged_table['raw_table']
        
        extracted_values = {}

        min_page, _ = self.__get_page__(raw_table.bounding_regions)
        
        if table_type == 'LEGAL_CASES':
            if self.relevant_values.get('legal_cases') is None:
                self.relevant_values['legal_cases'] = True
        
        if table_type == 'LEGAL_CASES_SUMMARY':
            if self.relevant_values.get('legal_cases') is None:
                self.relevant_values['legal_cases'] = True
            val = self.__safe_get_cell__(table, -1, 0)
            last_idx_str = val.replace(".", "")
            if last_idx_str.isdigit():
                extracted_values['legal_cases_count'] = int(last_idx_str)    
                self.di_confidence['legal_cases_count'] = SearchContext(val, page_number=min_page)     
                  
        if table_type == 'TRADE_REFERENCE':
            if self.relevant_values.get('trade_reference') is None:
                self.relevant_values['trade_reference'] = True

        if table_type == 'TRADE_REFERENCE_SUMMARY':
            if self.relevant_values.get('trade_reference') is None:
                self.relevant_values['trade_reference'] = True
            val = self.__safe_get_cell__(table, -1, 0)
            last_idx_str = val.replace(".", "")
            if last_idx_str.isdigit():
                extracted_values['trade_reference_count'] = int(last_idx_str)    
                self.di_confidence['trade_reference_count'] = SearchContext(val, page_number=min_page)
         
        return extracted_values

    def __extract_from_table_individual__(self, tagged_table: Dict) -> Dict:
        """Extracts data from a single tagged table for individual report based on its type."""
        table_type = tagged_table['type']
        table = tagged_table['table']
        raw_table = tagged_table['raw_table']
        
        extracted_values = {}

        min_page, _ = self.__get_page__(raw_table.bounding_regions)
        
        if table_type == 'CREDIT_INFO_AT_A_GLANCE':
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()

                if self.__is_fuzzy_match__(row_key_text, 'legal records in past 24 months (non-personal capacity)', 100):
                    val = self.__safe_get_cell__(table, r_idx, 2)
                    if val is not None:
                        extracted_values['legal_non_personal'] = val
                        self.di_confidence['legal_non_personal'] = SearchContext(val, page_number=min_page)
                
                if self.__is_fuzzy_match__(row_key_text, 'legal records in past 24 months (personal capacity)', 100):
                    val = self.__safe_get_cell__(table, r_idx, 2)
                    if val is not None:
                        extracted_values['legal_personal'] = val
                        self.di_confidence['legal_personal'] = SearchContext(val, page_number=min_page)
                if self.__is_fuzzy_match__(row_key_text, 'special attention accounts'):
                    val = self.__safe_get_cell__(table, r_idx, 2)
                    if val is not None:
                        extracted_values['special_attention_accounts_0'] = val
                        self.di_confidence['special_attention_accounts_0'] = SearchContext(val, page_number=min_page)
                  
        elif table_type == 'CCRIS_SUMMARY':
            ccris_summary = self.__extract_from_ccris_summary__(table, raw_table)
            extracted_values.update(ccris_summary)
        
        elif table_type == 'CCRIS_DETAILS_SINGLE':
            ccris_details = self.__extract_from_ccris_details_single__(table, raw_table)
            extracted_values.update(ccris_details)

        elif table_type == 'CCRIS_DETAILS_MULTI_START':
            ccris_details = self.__extract_from_ccris_details_multi_start__(table, raw_table)
            extracted_values.update(ccris_details)
        
        elif table_type == 'CCRIS_DETAILS_MULTI_MID':
            ccris_details = self.__extract_from_ccris_details_multi_mid__(table, raw_table)
            extracted_values.update(ccris_details)

        elif table_type == 'CCRIS_DETAILS_MULTI_END':
            ccris_details = self.__extract_from_ccris_details_multi_end__(table, raw_table)
            extracted_values.update(ccris_details)
             
        return extracted_values

    def __extract_from_ccris_summary__(self, table, raw_table) -> Dict:
        """Extracts data from CCRIS Summary table."""
        extracted_values = {}
        min_page, _ = self.__get_page__(raw_table.bounding_regions)
        
        for r_idx in table:
            row_key_text = table[r_idx].get(0, "").strip().lower()
            if self.__is_fuzzy_match__(row_key_text, 'as borrower'):
                val1 = self.__safe_get_cell__(table, r_idx, 1)
                val2 = self.__safe_get_cell__(table, r_idx, 2)
                if val1 is not None:
                    extracted_values['total_outstanding_balance_0'] = val1
                    self.di_confidence['total_outstanding_balance_0'] = SearchContext(val1, page_number=min_page)
                if val2 is not None:
                    extracted_values['total_limit_0'] = val2
                    self.di_confidence['total_limit_0'] = SearchContext(val2, page_number=min_page)
            
            if self.__is_fuzzy_match__(row_key_text, 'special attention account'):
                val = self.__safe_get_cell__(table, r_idx, 1)
                if val is not None:
                    extracted_values['special_attention_accounts_1'] = val
                    self.di_confidence['special_attention_accounts_1'] = SearchContext(val, page_number=min_page)
        return extracted_values 
    
    def __extract_from_ccris_details_single__(self, table, raw_table) -> Dict:
        """Extracts data from CCRIS Details Single table."""
        extracted_values = {}
        end_row_idx = -1
        start_row_idx = -1
        min_page, _ = self.__get_page__(raw_table.bounding_regions)
        for r_idx in table:
            row_key_text = table[r_idx].get(5, "").strip().lower()
            if self.__is_fuzzy_match__(row_key_text, 'total outstanding balance'):
                val6 = self.__safe_get_cell__(table, r_idx, 6)
                val8 = self.__safe_get_cell__(table, r_idx, 8)
                if val6 is not None:
                    extracted_values['total_outstanding_balance_1'] = val6
                    self.di_confidence['total_outstanding_balance_1'] = SearchContext(val6, page_number=min_page)
                if val8 is not None:
                    extracted_values['total_limit_1'] = val8
                    self.di_confidence['total_limit_1'] = SearchContext(val8, page_number=min_page)
                end_row_idx = r_idx

        for r_idx in table:
            header_text = table[r_idx].get(0, "").strip().lower()
            if self.__is_fuzzy_match__(header_text, 'outstanding credit'):
                start_row_idx = r_idx + 1

        if (start_row_idx != -1 and end_row_idx != -1):
            conduct_values = self.__extract_conduct__(table, start_row_idx, end_row_idx, raw_table)
            extracted_values['ccris_conduct'] = conduct_values
        return extracted_values

    def __extract_from_ccris_details_multi_start__(self, table, raw_table) -> Dict:
        """Extracts data from CCRIS Details Multi Start table."""
        extracted_values = {}
        start_row_idx = -1
        for r_idx in table:
            header_text = table[r_idx].get(0, "").strip().lower()
            if self.__is_fuzzy_match__(header_text, 'outstanding credit'):
                start_row_idx = r_idx + 1

        if (start_row_idx != -1):
            conduct_values = self.__extract_conduct__(table, start_row_idx, len(table), raw_table)
            extracted_values['ccris_conduct'] = conduct_values
        return extracted_values
    
    def __extract_from_ccris_details_multi_mid__(self, table, raw_table) -> Dict:
        """Extracts data from CCRIS Details Multi Mid table."""
        extracted_values = {}
        conduct_values = self.__extract_conduct__(table, 0, len(table), raw_table)
        extracted_values['ccris_conduct'] = conduct_values
        return extracted_values
    
    def __extract_from_ccris_details_multi_end__(self, table, raw_table) -> Dict:
        """Extracts data from CCRIS Details Multi End table."""
        extracted_values = {}
        end_row_idx = -1
        min_page, _ = self.__get_page__(raw_table.bounding_regions)
        for r_idx in table:
            row_key_text = table[r_idx].get(5, "").strip().lower()
            if self.__is_fuzzy_match__(row_key_text, 'total outstanding balance'):
                val6 = self.__safe_get_cell__(table, r_idx, 6)
                val8 = self.__safe_get_cell__(table, r_idx, 8)
                if val6 is not None:
                    extracted_values['total_outstanding_balance_1'] = val6
                    self.di_confidence['total_outstanding_balance_1'] = SearchContext(val6, page_number=min_page)
                if val8 is not None:
                    extracted_values['total_limit_1'] = val8
                    self.di_confidence['total_limit_1'] = SearchContext(val8, page_number=min_page)
                end_row_idx = r_idx

        if (end_row_idx != -1):
            conduct_values = self.__extract_conduct__(table, 0, end_row_idx, raw_table)
            extracted_values['ccris_conduct'] = conduct_values
        return extracted_values
    
    def __extract_from_table_company__(self, tagged_table: Dict) -> Dict:
        """Extracts data from a single tagged table for company report based on its type."""
        table_type = tagged_table['type']
        table = tagged_table['table']
        raw_table = tagged_table['raw_table']
        
        extracted_values = {}

        min_page, _ = self.__get_page__(raw_table.bounding_regions)
        
        if table_type == 'CREDIT_INFO_AT_A_GLANCE':
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()

                if self.__is_fuzzy_match__(row_key_text, 'legal records in past 24 months (non-personal capacity)', 100):
                    val2 = self.__safe_get_cell__(table, r_idx, 2)
                    val3 = self.__safe_get_cell__(table, r_idx, 3)
                    if val2 is not None:
                        extracted_values['legal_non_personal_entity'] = val2
                        self.di_confidence['legal_non_personal_entity'] = SearchContext(val2, page_number=min_page)
                    if val3 is not None:
                        extracted_values['legal_non_personal_rp'] = val3
                        self.di_confidence['legal_non_personal_rp'] = SearchContext(val3, page_number=min_page)
                
                if self.__is_fuzzy_match__(row_key_text, 'legal records in past 24 months (personal capacity)', 100):
                    val2 = self.__safe_get_cell__(table, r_idx, 2)
                    val3 = self.__safe_get_cell__(table, r_idx, 3)
                    if val2 is not None:
                        extracted_values['legal_personal_entity'] = val2
                        self.di_confidence['legal_personal_entity'] = SearchContext(val2, page_number=min_page)
                    if val3 is not None:
                        extracted_values['legal_personal_rp'] = val3
                        self.di_confidence['legal_personal_rp'] = SearchContext(val3, page_number=min_page)
                   
                if self.__is_fuzzy_match__(row_key_text, 'special attention accounts'):
                    val2 = self.__safe_get_cell__(table, r_idx, 2)
                    val3 = self.__safe_get_cell__(table, r_idx, 3)
                    if val2 is not None:
                        extracted_values['special_attention_accounts_entity'] = val2
                        self.di_confidence['special_attention_accounts_entity'] = SearchContext(val2, page_number=min_page)
                    if val3 is not None:
                        extracted_values['special_attention_accounts_rp'] = val3
                        self.di_confidence['special_attention_accounts_rp'] = SearchContext(val3, page_number=min_page)
                  
        elif table_type == 'CCRIS_SUMMARY':
            ccris_summary = self.__extract_from_ccris_summary__(table, raw_table)
            extracted_values.update(ccris_summary)
        
        elif table_type == 'CCRIS_DETAILS_SINGLE':
            ccris_details = self.__extract_from_ccris_details_single__(table, raw_table)
            extracted_values.update(ccris_details)

        elif table_type == 'CCRIS_DETAILS_MULTI_START':
            ccris_details = self.__extract_from_ccris_details_multi_start__(table, raw_table)
            extracted_values.update(ccris_details)
        
        elif table_type == 'CCRIS_DETAILS_MULTI_MID':
            ccris_details = self.__extract_from_ccris_details_multi_mid__(table, raw_table)
            extracted_values.update(ccris_details)

        elif table_type == 'CCRIS_DETAILS_MULTI_END':
            ccris_details = self.__extract_from_ccris_details_multi_end__(table, raw_table)
            extracted_values.update(ccris_details)
       
        elif table_type == 'SNAPSHOT':
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()
                if self.__is_fuzzy_match__(row_key_text, 'registration date'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['registration_date'] = val
                        self.di_confidence['registration_date'] = SearchContext(val, page_number=min_page)
                
                if self.__is_fuzzy_match__(row_key_text, 'type', 100):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['type'] = " ".join(val.splitlines())
                        self.di_confidence['type'] = SearchContext(val, page_number=min_page)
                
                if self.__is_fuzzy_match__(row_key_text, 'type of company'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['type'] = " ".join(val.splitlines())
                        self.di_confidence['type'] = SearchContext(val, page_number=min_page)

                if self.__is_fuzzy_match__(row_key_text, 'msic'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['msic'] = " ".join(val.splitlines())
                        self.di_confidence['msic'] = SearchContext(val, page_number=min_page)

                if self.__is_fuzzy_match__(row_key_text, 'business commenced') or self.__is_fuzzy_match__(row_key_text, 'last changed date') or self.__is_fuzzy_match__(row_key_text, 'rob search date') or self.__is_fuzzy_match__(row_key_text, 'current registration expiry date'):
                    if self.relevant_values.get('partnership') is None:
                        self.relevant_values['partnership'] = True
        
        elif table_type == 'DIRECTORS_OFFICERS':
            director_count = 0
            for r_idx in table:
                row_key_text = table[r_idx].get(4, "").strip().lower()
                if self.__is_fuzzy_match__(row_key_text, 'ds'):
                    director_count += 1
            extracted_values['director_count'] = director_count
            self.di_confidence['director_count'] = SearchContext('DS', page_number=min_page)
        
        elif table_type == 'BUSINESS_PROFILE':
            if self.relevant_values.get('partnership') is None:
                self.relevant_values['partnership'] = True
            partner_count = 0
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()
                if self.__is_fuzzy_match__(row_key_text, 'position'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None and self.__is_fuzzy_match__(val, 'partner'):
                        partner_count += 1
            extracted_values['partner_count'] = partner_count
            self.di_confidence['partner_count'] = SearchContext('PARTNER', page_number=min_page)

        elif table_type == 'FINANCIALS_AND_SHAREHOLDERS':
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()
                if self.__is_fuzzy_match__(row_key_text, 'revenue (rm)'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['revenue_0'] = val
                        self.di_confidence['revenue_0'] = SearchContext(val, page_number=min_page)
                if self.__is_fuzzy_match__(row_key_text, 'profit after tax (rm)'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['profit_after_tax_0'] = val
                        self.di_confidence['profit_after_tax_0'] = SearchContext(val, page_number=min_page)
                if self.__is_fuzzy_match__(row_key_text, 'paid up capital (rm)'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['paid_up_capital'] = val
                        self.di_confidence['paid_up_capital'] = SearchContext(val, page_number=min_page)

        elif table_type == 'FINANCIAL_STATEMENTS':
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()
                if self.__is_fuzzy_match__(row_key_text, 'financial year end'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['financial_year_end'] = val
                        self.di_confidence['financial_year_end'] = SearchContext(val, page_number=min_page)
                
                if self.__is_fuzzy_match__(row_key_text, 'non-current assets', 100):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['non_current_assets'] = val
                        self.di_confidence['non_current_assets'] = SearchContext(val, page_number=min_page)
                    
                if self.__is_fuzzy_match__(row_key_text, 'current assets', 100):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['current_assets'] = val
                        self.di_confidence['current_assets'] = SearchContext(val, page_number=min_page)
                    
                if self.__is_fuzzy_match__(row_key_text, 'total assets', 100):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['total_assets'] = val
                        self.di_confidence['total_assets'] = SearchContext(val, page_number=min_page)

                if self.__is_fuzzy_match__(row_key_text, 'non-current liabilities', 100):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['non_current_liabilities'] = val
                        self.di_confidence['non_current_liabilities'] = SearchContext(val, page_number=min_page)
                
                if self.__is_fuzzy_match__(row_key_text, 'current liabilities', 100):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['current_liabilities'] = val
                        self.di_confidence['current_liabilities'] = SearchContext(val, page_number=min_page)
                
                if self.__is_fuzzy_match__(row_key_text, 'long term liabilities', 100):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['long_term_liabilities'] = val
                        self.di_confidence['long_term_liabilities'] = SearchContext(val, page_number=min_page)

                if self.__is_fuzzy_match__(row_key_text, 'total liabilities', 100):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['total_liabilities'] = val
                        self.di_confidence['total_liabilities'] = SearchContext(val, page_number=min_page)
                     
                if self.__is_fuzzy_match__(row_key_text, 'retained earning'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['retained_earning'] = val
                        self.di_confidence['retained_earning'] = SearchContext(val, page_number=min_page)

                if self.__is_fuzzy_match__(row_key_text, 'net worth (ta - tl)'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['net_worth'] = val
                        self.di_confidence['net_worth'] = SearchContext(val, page_number=min_page)
                
                if self.__is_fuzzy_match__(row_key_text, 'revenue'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['revenue_1'] = val
                        self.di_confidence['revenue_1'] = SearchContext(val, page_number=min_page)

                if self.__is_fuzzy_match__(row_key_text, 'profit / (loss) after tax', 100):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['profit_after_tax_1'] = val
                        self.di_confidence['profit_after_tax_1'] = SearchContext(val, page_number=min_page)

                if self.__is_fuzzy_match__(row_key_text, 'current ratio'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['current_ratio'] = val
                        self.di_confidence['current_ratio'] = SearchContext(val, page_number=min_page)

                if self.__is_fuzzy_match__(row_key_text, 'gearing ratio'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['gearing_ratio'] = val
                        self.di_confidence['gearing_ratio'] = SearchContext(val, page_number=min_page)

                if self.__is_fuzzy_match__(row_key_text, 'debt to equity ratio [%]'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['debt_to_equity_ratio'] = val
                        self.di_confidence['debt_to_equity_ratio'] = SearchContext(val, page_number=min_page)

        return extracted_values

    def __map_parsed_data__(self, parsed_data: Dict):
        """Maps parsed data keys to final output keys."""
        mapped_data = {}

        mapped_data['report_type'] = self.report_type.value
        
        if self.relevant_values.get('financial_statements') is not None and self.relevant_values.get('financial_statements') == True:
            mapped_data['financial_report_provided'] = 'YES'
        else:
            mapped_data['financial_report_provided'] = 'NO'

        for key, value in parsed_data.items():

            if key == 'repayment_to_banks':
                # override if there is SPA
                if parsed_data.get('special_attention_accounts') is not None and parsed_data.get('special_attention_accounts') == 'YES':
                    mapped_data['repayment_to_banks'] = 'Unsatisfactory ( Under SPA or consistently lapsed 2 months and above )'
                else:
                    mapped_data['repayment_to_banks'] = value

            elif key == 'utilisation':
                if value == 'N/A':
                    mapped_data['utilisation'] = 'N/A'
                else:
                    value = self.__to_decimal__(value)
                    if value == 0:
                        if self.report_type == ReportType.INDIVIDUAL:
                            mapped_data['utilisation'] = '0%'
                        else:
                            mapped_data['utilisation'] = '0% ( No outstanding balance )'
                    elif 1 <= value <= 25:
                        mapped_data['utilisation'] = '( 1% - 25% )'
                    elif 26 <= value <= 50:
                        mapped_data['utilisation'] = '( 26% - 50% )'
                    elif 51 <= value <= 75:
                        mapped_data['utilisation'] = '( 51% - 75% )'
                    elif 76 <= value <= 100:
                        mapped_data['utilisation'] = '( 76% - 100% )'
                    else:
                        mapped_data['utilisation'] = 'N/A'

            elif key == 'special_attention_accounts':
                mapped_data['special_attention_accounts'] = value

            elif key == 'legal_cases':
                value = int(value)
                if value == 0:
                    mapped_data['legal_cases'] = '0 ( Clean of legal action )'
                elif value == 1:
                    mapped_data['legal_cases'] = '1 ( 1 case still on-going or unsettled )'
                elif value == 2:
                    mapped_data['legal_cases'] = '2 ( 2 cases still on-going or unsettled )'
                elif value == 3:
                    mapped_data['legal_cases'] = '3 ( 3 cases still on-going or unsettled )'
                else:
                    mapped_data['legal_cases'] = '>3 ( More than 3 cases still on-going or unsettled )'

            elif key == 'blacklist':
                value = int(value)
                if value == 0:
                    mapped_data['blacklist_cases'] = '0 ( no blacklist issue )'
                elif value == 1:
                    mapped_data['blacklist_cases'] = '1 ( 1 blacklist issue )'
                elif value == 2:
                    mapped_data['blacklist_cases'] = '2 ( 2 blacklist issue )'
                elif value == 3:
                    mapped_data['blacklist_cases'] = '3 ( 3 blacklist issue )'
                else:
                    mapped_data['blacklist_cases'] = '>3 ( More than 3 blacklist issue )'
                    
            elif key == 'years_in_business':
                value = int(value)
                if value < 2:
                    mapped_data['years_in_business'] = '< 2 Years'
                elif 2 <= value <= 5:
                    mapped_data['years_in_business'] = '2.1 - 5 Years'
                elif 5 < value <= 7:
                    mapped_data['years_in_business'] = '5.1 - 7 Years'
                elif 7 < value <= 10:
                    mapped_data['years_in_business'] = '7.1 - 10 years'
                else:
                    mapped_data['years_in_business'] = '> 10 Years'

            elif key == 'type_of_company':
                mapped_data['type_of_company'] = value

            elif key == 'nature_of_business':
                # TODO map MSIC to category
                mapped_data['nature_of_business'] = value

            elif key == 'number_of_directors_or_partners':
                mapped_data['number_of_directors_or_partners'] = value

            elif key == 'paid_up_capital':
                # partnership (or non sdn bhd) form does not need paid_up_capital
                if value == 'N/A':
                    mapped_data['paid_up_capital'] = 'N/A'
                else:
                    value = self.__to_decimal__(value)
                    if value < 2:
                        mapped_data['paid_up_capital'] = '< 2K'
                    elif 2 <= value <= 150999:
                        mapped_data['paid_up_capital'] = '2K - 150K'
                    elif 151000 <= value <= 300999:
                        mapped_data['paid_up_capital'] = '151K - 300K'
                    elif 301000 <= value <= 750000:
                        mapped_data['paid_up_capital'] = '301K - 750K'
                    else:
                        mapped_data['paid_up_capital'] = '> 750K'

            elif key == 'financial_report_date':
                mapped_data['financial_report_date'] = value

            elif key == 'turnover':
                if value == 'N/A':
                    mapped_data['turnover_amount'] = value
                    mapped_data['turnover'] = 'Negative'
                else:
                    value = self.__to_decimal__(value)
                    mapped_data['turnover_amount'] = value
                    if value > 0:
                        mapped_data['turnover'] = 'Positive'
                    else:
                        mapped_data['turnover'] = 'Negative'

            elif key == 'net_profit':
                if value == 'N/A':
                    mapped_data['net_profit_amount'] = value
                    mapped_data['net_profit'] = 'Negative or N/A ( Make loss company or Not available)'
                else:
                    value = self.__to_decimal__(value)
                    mapped_data['net_profit_amount'] = value
                    if value < 0:
                        mapped_data['net_profit'] = 'Negative or N/A ( Make loss company or Not available)'
                    elif 0 <= value <= 300999:
                        mapped_data['net_profit'] = '0 - 300K'
                    elif 301000 <= value <= 500999:
                        mapped_data['net_profit'] = '301K - 500K'
                    elif 501000 <= value <= 999999:
                        mapped_data['net_profit'] = '501K - 999K'
                    else:
                        mapped_data['net_profit'] = '> 1 Mil'
                
            elif key == 'retained_profit':
                if value == 'N/A':
                    mapped_data['retained_profit_amount'] = value
                    mapped_data['retained_profit'] = 'Negative or N/A ( Making accumulated losses or Not available )'
                else:
                    value = self.__to_decimal__(value)
                    mapped_data['retained_profit_amount'] = value
                    if value < 0:
                        mapped_data['retained_profit'] = 'Negative or N/A ( Making accumulated losses or Not available )'
                    else:
                        mapped_data['retained_profit'] = 'Positive'

            elif key == 'net_worth':
                if value == 'N/A':
                    mapped_data['net_worth_amount'] = value
                    mapped_data['net_worth'] = 'Negative or N/a ( Is an insolvent company or Not available)'
                else:
                    value = self.__to_decimal__(value)
                    mapped_data['net_worth_amount'] = value
                    if value < 0:
                        mapped_data['net_worth'] = 'Negative or N/a ( Is an insolvent company or Not available)'
                    else:
                        mapped_data['net_worth'] = 'Positive'

            elif key == 'net_current_assets':
                if value == 'N/A':
                    mapped_data['net_current_assets_amount'] = value
                    mapped_data['net_current_assets'] = '( Negative working capital / Not available )'
                else:
                    value = self.__to_decimal__(value)
                    mapped_data['net_current_assets_amount'] = value
                    if value < 0:
                        mapped_data['net_current_assets'] = '( Negative working capital / Not available )'
                    else:
                        mapped_data['net_current_assets'] = 'Positive'

            elif key == 'current_ratio':
                if value == 'N/A':
                    mapped_data['current_ratio'] = 'N/A'
                else:
                    value = self.__to_decimal__(value)
                    if value < 1:
                        mapped_data['current_ratio'] = '< 1.00'
                    elif 1 <= value <= Decimal('1.99'):
                        mapped_data['current_ratio'] = '1.01 - 1.99'
                    elif value >= 2:
                        mapped_data['current_ratio'] = '> 2.00'
                    else:
                        mapped_data['current_ratio'] = 'N/A'
                
            elif key == 'gearing_ratio':
                if value == 'N/A':
                    mapped_data['gearing_ratio'] = 'N/A'
                else:
                    value = self.__to_decimal__(value)
                    if value < 0:
                        mapped_data['gearing_ratio'] = 'Negative'
                    elif 0 <= value <= Decimal('0.99'):
                        mapped_data['gearing_ratio'] = '( 0 - 0.99 )'
                    elif 1 <= value <= Decimal('1.99'):
                        mapped_data['gearing_ratio'] = '( 1.00 - 1.99 )'
                    elif 2 <= value <= Decimal('2.99'):
                        mapped_data['gearing_ratio'] = '( 2.00 - 2.99 )'
                    elif 3 <= value <= Decimal('3.99'):
                        mapped_data['gearing_ratio'] = '( 3.00 - 3.99 )'
                    elif value >= 4:
                        mapped_data['gearing_ratio'] = '> 4.00'
                    else:
                        mapped_data['gearing_ratio'] = 'N/A'
            
        return mapped_data

    def __get_page_range_for_table_tag__(self, table_tag: str) -> Tuple[Optional[int], Optional[int]]:
        """Returns the page range (start, end) for a given table tag."""
        # Map extract_using_image tags to the table type constants used during tagging
        tag_to_types = {
            'ccris_summary': ['CCRIS_SUMMARY'],
            'ccris_detail': ['CCRIS_DETAILS_SINGLE', 'CCRIS_DETAILS_MULTI_START', 'CCRIS_DETAILS_MULTI_MID', 'CCRIS_DETAILS_MULTI_END'],
            'ccris_detail_edge_case': ['CCRIS_DETAILS_SINGLE'],
            'credit_info_at_a_glance': ['CREDIT_INFO_AT_A_GLANCE'],
            'snapshot': ['SNAPSHOT'],
            'financials_and_shareholders': ['FINANCIALS_AND_SHAREHOLDERS'],
            'financial_statements': ['FINANCIAL_STATEMENTS'],
            'directors_officers': ['DIRECTORS_OFFICERS'],
            'business_profile': ['BUSINESS_PROFILE'],
            'trade_reference': ['TRADE_REFERENCE']
        }

        if table_tag == 'ccris_detail' or table_tag == 'ccris_detail_edge_case' or table_tag == 'trade_reference':
            min_page = self.table_page_ranges.get('CCRIS_SUMMARY', (0, len(self.result.pages)))[0]
            max_page = len(self.result.pages)
            return (min_page, max_page)
        
        types = tag_to_types.get(table_tag, [])
        min_page, max_page = None, None
        
        for t in types:
            if t in self.table_page_ranges:
                t_min, t_max = self.table_page_ranges[t]
                min_page = t_min if min_page is None else min(min_page, t_min)
                max_page = t_max if max_page is None else max(max_page, t_max)

        return (min_page, max_page)
            
    def __get_prompt_for_table_tag__(self, table_tag: str) -> str:
        """Returns the prompt string for a given table tag."""
        match table_tag:
            case 'trade_reference':
                return (
                    "For the section 'E2: TRADE REFERENCE', "
                    "check if there is 'No Information Available' below the section heading. "
                    "If 'No Information Available' appears, return false for 'has_trade_reference'. "
                    "If there are tables under this section with subheadings like 'The following information are in relation to Account No' "
                    "or 'Aging Information', return true for 'has_trade_reference' and count the number of distinct trade reference entries "
                    "in the summary table as 'trade_reference_count'. Return the extracted data in the following JSON format: {\"has_trade_reference\": value, \"trade_reference_count\": value}."
                )
            case 'directors_officers':
                if self.relevant_values.get('partnership') is None:
                    return (
                        "Extract the number of directors from the table with the heading 'DIRECTORS / OFFICERS'. The column 'Designation' indicates the status for each row, where 'DS' indicates a director. Count the number of occurrences of 'DS' in the 'Designation' column to determine the number of directors. If there are no directors listed, return 0. Return the extracted data in the following JSON format: {\"director_count\": value}."
                    )
                else:
                    return ("") 
            case 'business_profile':
                if self.relevant_values.get('partnership') is not None:
                    return (
                        "Extract the number of partners from the table with the heading 'B1: BUSINESS PROFILE'. The table shows the personal details such as name, id, status, position for each partner or owner. The row 'Position' indicates the position of the individual, where 'Partner' indicates a partner. If there are no partners listed, return 0. Return the extracted data in the following JSON format: {\"partner_count\": value}."
                    )
                else:
                    return ("")
            case 'ccris_summary':
                return (
                    "Extract the following fields from the table with the heading 'C1: BANKING PAYMENT RECORDS (SOURCE: CCRIS, BANK NEGARA MALAYSIA)'. Under the subheading 'Summary of Potential & Current Liabilities', for the first row labeled 'As Borrower', extract the two values of total outstanding balance and total limit from the columns 'Outstanding' and 'Total Limit'. If the value is 0, it may be represented as a dash '-' or an en-dash '–' or an em-dash '—'. If the value is 0.00, return 0.00 not null. Brackets surrounding a numerical value indicates that the numerical value is negative. Extract the value ('Y' or 'N') for the field 'Special Attention Account' which is the last row of the table, under the column 'Outstanding'. If any of these fields are not present in the table, return null for that field. Return the extracted data in the following JSON format: {\"total_outstanding_balance\": value or null, \"total_limit\": value or null, \"special_attention_accounts\": value or null}."
                )
            case 'ccris_detail':
                return (
                    "Attached are images of pages from a credit report containing a table with the heading 'CCRIS Details' and subheadings 'Loan Information', 'Special Attention Account', and 'Credit Application'. The columns are: 'No', 'Date', 'Sts', 'Capacity', 'Lender Type', 'Facility', 'Total Outstanding Balance', 'Data Balance Updated', 'Limit/Installment Amount', 'Prin. Repmt. Term', 'Col Type', 'Conduct of Account For Last 12 Months', 'LGL STS', and 'Date Status Updated'. The column 'Conduct of Account For Last 12 Months' contains 12 sub-columns representing the repayment conduct for each of the last 12 months, with values representing the number of months the payment was late (0 for on-time payment). We are only interested in extracting data from the column 'Conduct of Account For Last 12 Months' and the summary row showing 'Total Outstanding Balance' and 'Total Limit' right before the subheading 'Special Attention Account'. "
                    "The 'CCRIS Details' table may span multiple pages. "
                    "The 'CCRIS Details' table ends when you encounter the 'Remark Legend', or any section header that is clearly not part of the CCRIS details table. "
                    "Extract the following from the CCRIS details table: "
                    "1. 'total_outstanding_balance': The total outstanding balance value from the summary row at the bottom of the table, right before the subheading 'Special Attention Account'. "
                    "2. 'total_limit': The total limit value from the summary row at the bottom of the table, right before the subheading 'Special Attention Account'. "
                    "3. 'ccris_conduct': For each loan row, extract the values (the numeric digits in the monthly columns under the column 'Conduct of Account For Last 12 Months'). There may be multiple loan rows. For each loan row, collect the values into a list of integers, for example [[0,0,1,0,0,0,0,0,2,0,0,0], [0,0,1,0,0,0,0,0,2,0,0,0]] for two loan rows. A loan row is indicated by an 'O' under the column 'Sts'. Do not represent a missing month with a 0, just skip it. A non-zero digit is usually in a shaded or colored cell. A digit which is zero is usually in an unshaded or uncolored cell. If you are unsure of the individual digits extracted, then return null for ccris_conduct. Ensure that all loan rows are extracted, with reference to the markdown table, the images provided, and this OCR result from another tool (which may contain errors such as 'O' or 'D' for zeroes, so use with caution): {self.di_conduct} ). "
                    "Return the extracted data in the following JSON format: "
                    "{\"total_outstanding_balance\": value or null, \"total_limit\": value or null, \"ccris_conduct\": [list of conduct strings] or null}."
                )
            case 'ccris_detail_edge_case':
                return (
                    "Attached are images of pages from a credit report containing a table with the heading 'CCRIS Details' and subheadings 'Loan Information', 'Special Attention Account', and 'Credit Application'. The columns are: 'No', 'Date', 'Sts', 'Capacity', 'Lender Type', 'Facility', 'Total Outstanding Balance', 'Data Balance Updated', 'Limit/Installment Amount', 'Prin. Repmt. Term', 'Col Type', 'Conduct of Account For Last 12 Months', 'LGL STS', and 'Date Status Updated'. The column 'Conduct of Account For Last 12 Months' contains 12 sub-columns representing the repayment conduct for each of the last 12 months, with values representing the number of months the payment was late (0 for on-time payment). We are only interested in extracting data from the column 'Conduct of Account For Last 12 Months' and the summary row showing 'Total Outstanding Balance' and 'Total Limit' right before the subheading 'Special Attention Account'. "
                    "The 'CCRIS Details' table may span multiple pages. "
                    "The 'CCRIS Details' table ends when you encounter the 'Remark Legend', or any section header that is clearly not part of the CCRIS details table. "
                    "Extract the following from the CCRIS details table: "
                    "1. 'total_outstanding_balance': The total outstanding balance value from the summary row at the bottom of the table, right before the subheading 'Special Attention Account'. "
                    "2. 'total_limit': The total limit value from the summary row at the bottom of the table, right before the subheading 'Special Attention Account'. "
                    "3. 'ccris_conduct': For each loan row, extract the values (the numeric digits in the monthly columns under the column 'Conduct of Account For Last 12 Months'). There may be multiple loan rows. For each loan row, collect the values into a list of integers, for example [[0,0,1,0,0,0,0,0,2,0,0,0], [0,0,1,0,0,0,0,0,2,0,0,0]] for two loan rows. A loan row is indicated by an 'O' under the column 'Sts'. Do not represent a missing month with a 0, just skip it. A non-zero digit is usually in a shaded or colored cell. A digit which is zero is usually in an unshaded or uncolored cell. If you are unsure of the individual digits extracted, then return null for ccris_conduct. Ensure that all loan rows are extracted, with reference to the markdown table, the images provided, and this OCR result from another tool (which may contain errors such as 'O' or 'D' for zeroes, so use with caution): {self.di_conduct} ). "
                    "Return the extracted data in the following JSON format: "
                    "{\"total_outstanding_balance\": value or null, \"total_limit\": value or null, \"ccris_conduct\": [list of conduct strings] or null}."
                )
            case 'credit_info_at_a_glance':
                if self.report_type == ReportType.INDIVIDUAL:
                    return (
                        "Extract the following fields from the table with the heading 'Credit Info at a Glance'. There are three columns: 'Credit Info', 'Source', 'Value'. We are only interested in the first column which shows the field names, and the third column 'Value' which shows the values for the entity. Extract the number of legal records in past 24 months (personal capacity) which is the first subrow for the field 'legal records in past 24 months (personal capacity)'. Extract the number of legal records in past 24 months (non-personal capacity) which is the first subrow for the field 'legal records in past 24 months (non-personal capacity)'. Extract the value for the field 'Special Attention Accounts'. If any of these fields are not present in the table, return null for that field. Return the extracted data in the following JSON format: {\"legal_personal\": value or null, \"legal_non_personal\": value or null, \"special_attention_accounts\": value or null}."
                    )
                elif self.report_type == ReportType.COMPANY:
                    return (
                        "Extract the following fields from the table with the heading 'Credit Info at a Glance'. There are four columns: 'Credit Info', 'Source', 'Entity', 'Related Parties'. We are only interested in the first column which shows the field names, and the third column 'Entity' which shows the values for the entity. Extract the Entity's number of legal records in past 24 months (personal capacity) which is the first subrow for the field 'legal records in past 24 months (personal capacity)'. Extract the Entity's number of legal records in past 24 months (non-personal capacity) which is the first subrow for the field 'legal records in past 24 months (non-personal capacity)'. Extract the Entity value for the field 'Special Attention Accounts'. If any of these fields are not present in the table, return null for that field. Return the extracted data in the following JSON format: {\"legal_personal_entity\": value or null, \"legal_non_personal_entity\": value or null, \"special_attention_accounts_entity\": value or null}."
                    )
            case 'snapshot':
                return (
                    "Extract the following fields from the table with the heading 'A: SNAPSHOT'. Extract the value for the field 'Registration Date'. If you see any field labelled 'type' or 'type of company', extract the value for the field. Extract the value for the field 'MSIC'. If any of these fields are not present in the table, return null for that field. If you see any field labelled 'business commenced' or 'last changed date' or 'rob search date' or 'current registration expiry date' in the table, then this table is for a partnership and you should return true for is_partnership. Return the extracted data in the following JSON format: {\"registration_date\": value or null, \"type\": value or null, \"msic\": value or null, \"is_partnership\": true or false}."
                )
            case 'financials_and_shareholders':
                return (
                    "Extract the value of 'Paid-Up Capital (RM)' from the table with the heading 'Financials and Shareholders'. The first column of the table is the field name and the second column of the table is the value. If this field is not present in the table, return null for that field. If the value is 0, it may be represented as a dash '-' or an en-dash '–' or an em-dash '—'. If the value is 0.00, return 0.00 and do not return null. Brackets surrounding a numerical value indicates that the numerical value is negative. Return the extracted data in the following JSON format: {\"paid_up_capital\": value or null}."
                )
            case 'financial_statements':
                return (
                    "Attached are images of financial statements of a company. Each financial statement is a table. " 
                    "Each table has six columns, with the first column being the financial item and the second column being the value for the latest financial year. "
                    "We are only interested in the values for the latest financial year in the second column. "
                    "Extract the following fields for the latest financial year (second column) from the tables. "
                    "From the header, extract 'financial year end' in YYYY-MM-DD format. "
                    "From the balance sheet, extract 'non-current assets', 'current assets', 'total assets', 'non-current liabilities', 'current liabilities', 'long term liabilities', 'total liabilities'. "
                    "From the income statement, extract 'revenue', 'profit / (loss) after tax'. "
                    "From the liquidity ratios, extract 'current ratio'. "
                    "From the leverage ratios, extract 'gearing ratio' and 'debt to equity ratio [%]'. "
                    "The values are in two decimal places. Brackets surrounding a numerical value indicates that the numerical value is negative. "
                    "Ignore asterisks around values if present. If the value is 0, it may be represented as a dash '-' or an en-dash '–' or an em-dash '—'. IMPORTANT: If a field's value is shown as a dash '-' or an en-dash '–' or an em-dash '—', this represents zero (0.00). You MUST return 0.00 for that field, NOT null. "
                    "Only return null if the field name (row) itself is completely missing from all the tables in all the images attached. If the row exists but shows a dash, return 0.00. "
                    "Due to OCR errors, some commas may be represented as periods, and vice versa. Always treat the right-most separator as the decimal point if ambiguous. "
                    "If any of these fields are not present in the tables, return null for that field. "
                    "Return the extracted data in the following JSON format: "
                    "{\"financial_year_end\": value or null, \"non_current_assets\": value or null, \"current_assets\": value or null, \"total_assets\": value or null, \"non_current_liabilities\": value or null, \"current_liabilities\": value or null, \"long_term_liabilities\": value or null, \"total_liabilities\": value or null, \"revenue\": value or null, \"profit_after_tax\": value or null, \"current_ratio\": value or null, \"gearing_ratio\": value or null, \"debt_to_equity_ratio\": value or null}."
                )
            case _:
                raise ValueError(f"Unknown table tag: {table_tag}")

    def __reformat_date__(self, date_str: str) -> str:
        """Reformats date string DD-MM-YYYY to YYYY-MM-DD."""
        try:
            date = datetime.strptime(date_str, '%d-%m-%Y')
            return date.strftime('%Y-%m-%d')
        except ValueError:
            return ""
    
    def __calculate_years__(self, date_str: str) -> float:
        """Calculates years (as a float) since date string DD-MM-YYYY."""
        try:
            past_date = datetime.strptime(date_str, '%d-%m-%Y')
            today = datetime.today()
            
            difference = today - past_date
            
            # 365.25 accounts for leap year cycles
            years_elapsed = difference.days / 365.25
            
            return round(years_elapsed, 2)
        except (ValueError, TypeError):
            return None
    
    def __str_to_decimal__(self, value: str) -> Decimal:
        """Converts a string representation of a number to Decimal, handling commas, spaces, and special characters."""
        try:
            # Handle nil/dash values
            stripped = value.strip()
            if stripped == '-' or stripped == '–' or stripped == '—' or stripped == '':
                return Decimal(0)
            
            clean_value = stripped.replace(',', '.').replace(' ', '').replace('%', '').replace('*', '')
            
            # Handle multiple periods (OCR errors)
            parts = clean_value.split('.')
            if len(parts) > 2:
                clean_value = ''.join(parts[:-1]) + '.' + parts[-1]
            
            if not clean_value:
                logger.warning("Empty value passed to __str_to_decimal__")
                return Decimal(0)
            if clean_value.startswith('(') and clean_value.endswith(')'):
                clean_value = '-' + clean_value[1:-1]

            result = Decimal(clean_value)
            
            if '%' in value:
                return result / 100
            return result
        except (InvalidOperation, AttributeError) as e:
            logger.warning("Failed to parse '%s' as Decimal: %s", value, e)
            return Decimal(0)
        
    def __parse_conduct_values_image__(self, conduct_values: List[List[int]]) -> str:
        """Evaluate conduct of account based on conduct values extracted from CCRIS Details table in image extraction."""
        flat_list = [item for sublist in conduct_values for item in sublist]
        zeroes = 0
        ones = 0
        twos = 0
        high_non_zeroes = 0
        digits = len(flat_list)
        for digit in flat_list:
            if digit == 0:
                zeroes += 1
            elif digit == 1:
                ones += 1
            elif digit == 2:
                twos += 1
            elif digit >= 3:
                high_non_zeroes += 1
            else:
                digits -= 1  # invalid
        non_zeroes = digits - zeroes
        if digits == zeroes or ((non_zeroes / digits) < 0.2 and non_zeroes == ones):
            return 'Satisfactory ( Prompt payment or occasionally lapsed 1 month )'
        elif (non_zeroes / digits) < 0.3 and non_zeroes == (ones + twos):
            return 'Moderate ( Consistently lapsed 1-2 months )'
        else:
            return 'Unsatisfactory ( Under SPA or consistently lapsed 2 months and above )'

    def __parse_conduct_values__(self, conduct_values: List[str]) -> str:
        """Evaluate conduct of account based on conduct values extracted from CCRIS Details table."""
        digits = 0
        zeroes = 0
        non_zeroes = 0
        ones = 0
        twos = 0
        # This is assuming guarantor will never have >9 months lapses in payments, hence the gpt4o cross-checks are essential.
        high_non_zeroes = 0
        for string in conduct_values:
            for char in string:
                digits += 1
                if char == '0':
                    zeroes += 1
                else:
                    # check against char, not int(char) to avoid counting invalid characters (e.g. 'D', 'O') from OCR errors
                    non_zeroes += 1
                    if char == '1':
                        ones += 1
                    elif char == '2':
                        twos += 1
                    elif char in ['3', '4', '5', '6', '7', '8', '9']:
                        high_non_zeroes += 1
                    else:
                        non_zeroes -= 1
                        digits -= 1  # invalid character, do not count
        if digits == zeroes or ((non_zeroes / digits) < 0.2 and non_zeroes == ones):
            return 'Satisfactory ( Prompt payment or occasionally lapsed 1 month )'
        elif (non_zeroes / digits) < 0.3 and non_zeroes == (ones + twos):
            return 'Moderate ( Consistently lapsed 1-2 months )'
        else:
            return 'Unsatisfactory ( Under SPA or consistently lapsed 2 months and above )'

    def __extract_conduct__(self, table, start_row_idx: int, end_row_idx: int, raw_table) -> List[str]:
        """Extracts conduct information from CCRIS Details table."""
        conduct_values = []
        min_page, _ = self.__get_page__(raw_table.bounding_regions)
        for r_idx in range(start_row_idx, end_row_idx):
            for c_idx in range(11, 23):
                cell = table[r_idx].get(c_idx, "")
                if cell:
                    cell = re.sub(r'\s+', '', cell.strip()) # remove all whitespace
                    conduct_values.append(cell)
                    confidence_key = f'conduct_values_row_{r_idx}_col_{c_idx}'
                    self.di_confidence[confidence_key] = SearchContext(cell, page_number=min_page)
        return conduct_values

    def __get_openai_client__(self, options: DocumentDataExtractorOptions) -> AzureOpenAI:
        token_provider = get_bearer_token_provider(
            self.credential, "https://cognitiveservices.azure.com/.default")

        client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=options.openai_endpoint,
            azure_ad_token_provider=token_provider)

        return client

    def __get_document_intelligence_client__(self, options: DocumentDataExtractorOptions) -> DocumentIntelligenceClient:
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=options.doc_intelligence_endpoint,
            credential=self.credential
        )

        return document_intelligence_client

    def __get_document_image_uris__(self, document_bytes: bytes, page_start: Optional[int], page_end: Optional[int]) -> list:
        """Converts the specified document bytes to images using the pdf2image library and returns the image URIs.

        To call this method, poppler-utils must be installed on the system.
        """

        try:
            pages = convert_from_bytes(
                document_bytes,
                first_page=page_start,
                last_page=page_end,
                dpi=300
            )
            logger.debug("Converted PDF pages %s to %s to %d images", page_start, page_end, len(pages))
        except Exception as e:
            logger.error("PDF to image conversion failed: %s", e, exc_info=True)
            raise ValueError(f"Failed to convert PDF to images: {e}") from e

        image_uris = []
        
        for page in pages:
            byteIO = io.BytesIO()
            page.save(byteIO, format='PNG')
            base64_data = base64.b64encode(byteIO.getvalue()).decode('utf-8')
            image_uris.append(f"data:image/png;base64,{base64_data}")

        return image_uris