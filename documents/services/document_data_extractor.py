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
from shared.confidence.confidence_utils import merge_confidence_values
from shared.confidence.openai_confidence import evaluate_confidence as evaluate_confidence_openai
from shared.confidence.document_intelligence_confidence import evaluate_confidence as evaluate_confidence_di
from shared.confidence.confidence_result import ConfidenceResult, OVERALL_CONFIDENCE_KEY
import logging

logger = logging.getLogger(__name__)

class ReportType(enum.Enum):
    INDIVIDUAL = "INDIVIDUAL"
    COMPANY = "COMPANY"

ResponseFormatT = TypeVar(
    "ResponseFormatT"
)

ExtractionConfidenceResult = ConfidenceResult[ResponseFormatT | None]

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

        self.system_prompt = f"""You are an AI assistant that extracts data from specified tables in documents."""
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
        self.relevant_paras: Dict[str, int] = {}
        self.options: DocumentDataExtractorOptions = None
        self.bytes: bytes = None
        self.table_page_ranges: Dict[str, Tuple[int, int]] = {}

    def __safe_get_cell__(self, table, r_idx: int, c_idx: int) -> Optional[str]:
        """Safely gets and strips a cell value from a table row, returning None if the cell doesn't exist."""
        value = table[r_idx].get(c_idx)
        if value is None:
            logger.debug("Missing cell at row %d, col %d", r_idx, c_idx)
            return None
        return value.strip()

    '''
    def from_bytes(self, document_bytes: bytes, response_format: type[ResponseFormatT], options: DocumentDataExtractorOptions) -> ExtractionConfidenceResult:
        """Extracts structured data from the specified document bytes by converting the document to images and using an Azure OpenAI model to extract the data.

        :param document_bytes: The byte array content of the document to extract data from.
        :param options: The options for configuring the Azure OpenAI request for extracting data.
        :return: The structured data extracted from the document as a dictionary.
        """

        client = self.__get_openai_client__(options)
        di_client = self.__get_document_intelligence_client__(options)

        if options.page_start and options.page_end:
            page_range = f"{options.page_start}-{options.page_end}"
        else:
            page_range = None

        # For a more accurate extraction, we can use the Document Intelligence service to extract the document layout and convert it to markdown.
        if di_client:
            poller = di_client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=document_bytes,
                pages=page_range,
                output_content_format=DocumentContentFormat.MARKDOWN,
                content_type="application/pdf"
            )
            self.result: AnalyzeResult = poller.result()
            document_markdown = self.result.content
        else:
            document_markdown = None

        image_uris = self.__get_document_image_uris__(
            document_bytes, options.page_start, options.page_end)

        user_content = []
        user_content.append({
            "type": "text",
            "text": "placeholder"
        })

        if document_markdown:
            user_content.append({
                "type": "text",
                "text": document_markdown
            })

        for image_uri in image_uris:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_uri
                }
            })

        completion = client.beta.chat.completions.parse(
            model=options.deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": options.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            response_format=response_format,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            # Enabled to determine the confidence of the response.
            logprobs=True
        )

        response_obj = completion.choices[0].message.parsed
        response_obj_dict = response_obj.model_dump()

        confidence_openai = evaluate_confidence_openai(
            extract_result=response_obj_dict,
            choice=completion.choices[0]
        )

        if di_client:
            confidence_di = evaluate_confidence_di(
                extract_result=response_obj_dict,
                analyze_result=self.result
            )
            confidence = merge_confidence_values(
                confidence_a=confidence_di,
                confidence_b=confidence_openai
            )
        else:
            confidence = confidence_openai

        return ExtractionConfidenceResult(
            data=response_obj,
            confidence_scores=confidence,
            overall_confidence=confidence[OVERALL_CONFIDENCE_KEY]
        )
    '''

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

            self.relevant_paras.update(self.__find_paragraphs__())
            logger.info("Identified relevant paragraphs: %s", self.relevant_paras)

            self.report_type = self.__classify_report_type__()
            logger.info("Classified report type as: %s", self.report_type.value)

            tagged_tables = self.__identify_tables_from_json__()
            logger.info("Tagged %d relevant tables for extraction", len(tagged_tables))

            extracted_data = self.__extract_from_tagged_tables__(tagged_tables)
            logger.info("Extracted data: %s", extracted_data)

            parsed_data = self.__parse_extracted_data__(extracted_data)
            logger.info("Parsed extracted data: %s", parsed_data)
            logger.info("Completed extraction successfully")

        except Exception as e:
            logger.error("Extraction failed: %s", e, exc_info=True)
            raise

        # TODO if openai is low confidence, escalate to human review
        
        # Convert Decimal values to float for JSON serialization
        return {k: float(v) if isinstance(v, Decimal) else v for k, v in parsed_data.items()}
    
    def __normalize_numeric_str__(self, value: str) -> str:
        """Normalizes a numeric string to handle OCR errors.
        Handles cases where:
        - Commas are used as thousand separators: '2,091,202.00' -> '2091202.00'
        - Commas are misread as periods: '2,091.202.00' -> '2091202.00'
        - Comma is used as decimal separator (European format): '1,22' -> '1.22'
        """
        # Handle nil/dash values first
        stripped = value.strip()
        if stripped == '-' or stripped == '–' or stripped == '—' or stripped == '':
            return '0'
        
        # Remove all commas first (they're either thousand separators or OCR errors)
        stripped = stripped.replace(',', '')
        
        parts = stripped.split('.')
        if len(parts) <= 2:
            return stripped
        # Multiple periods: all but the last are OCR'd commas
        return ''.join(parts[:-1]) + '.' + parts[-1]

    # TODO evaluate necessity of this since false positives
    def __find_paragraphs__(self) -> Dict[str, int]:
        """Locate relevant paragraphs since information is captured either as paragraphs or tables."""
        relevant_paras = {}

        if not self.result.paragraphs:
            return relevant_paras
        
        num_paragraphs = len(self.result.paragraphs)
        
        for idx, para in enumerate(self.result.paragraphs):
            para_text = para.content.strip().lower()

            if idx + 1 >= num_paragraphs:
                continue
            
            if self.__is_fuzzy_match__(para_text, 'c1: banking payment records (source: ccris, bank negara malaysia)'):
                ccris = self.__find_paragraph_below_paragraph__(para, self.result.paragraphs)
                if ccris and self.__is_fuzzy_match__(ccris, 'a check with bank negara malaysia returned no result on subject', 98):
                    relevant_paras['ccris_not_available'] = idx
            if self.__is_fuzzy_match__(para_text, 'd1: legal cases (subject as defendant)'):
                defendant = self.__find_paragraph_below_paragraph__(para, self.result.paragraphs)
                if defendant and self.__is_fuzzy_match__(defendant, 'no information available', 98):
                    relevant_paras['legal_defendant_none'] = idx
            if self.__is_fuzzy_match__(para_text, 'd2: legal cases (subject as plaintiff)'):
                plaintiff = self.__find_paragraph_below_paragraph__(para, self.result.paragraphs)
                if plaintiff and self.__is_fuzzy_match__(plaintiff, 'no information available', 98):
                    relevant_paras['legal_plaintiff_none'] = idx
        return relevant_paras
    
    def __record_table_pages__(self, table_idx: int, table_tag: str):
        """Records the page range for a tagged table type."""
        table = self.result.tables[table_idx]
        pages = set()
        for region in (table.bounding_regions or []):
            pages.add(region.page_number)
        if not pages:
            return
        min_page, max_page = min(pages), max(pages)
        # Expand existing range if this tag was already seen (e.g. multi-page CCRIS_DETAILS_MULTI_MID tables)
        if table_tag in self.table_page_ranges:
            existing_min, existing_max = self.table_page_ranges[table_tag]
            min_page = min(existing_min, min_page)
            max_page = max(existing_max, max_page)
        self.table_page_ranges[table_tag] = (min_page, max_page)

    def __identify_tables_from_json__(self) -> List[Dict]:
        """Identify relevant tables."""
        
        tagged_tables = []
        
        if not self.result.tables:
            logger.warning("No tables found in document")
            return tagged_tables
        
        logger.info("Processing %d tables for tagging", len(self.result.tables))
        paragraphs = self.result.paragraphs or []
        
        for table_idx, table in enumerate(self.result.tables):
            table_region = table.bounding_regions[0] if table.bounding_regions else None
            
            # Find preceding paragraph to use as context
            preceding_text = self.__find_missing_header__(table_region, paragraphs)
            
            # Determine table type from headers or context
            idx_and_type = self.__determine_table_type__(table, preceding_text, table_idx)

            if len(idx_and_type) == 1:
                if idx_and_type[0]['type'] != 'UNKNOWN':
                    lookup_table = self.__convert_to_row_map__(table)   
                    tagged_tables.append({
                        'type': idx_and_type[0]['type'],
                        'table': lookup_table
                    })
                    self.__record_table_pages__(idx_and_type[0]['idx'], idx_and_type[0]['type'])
            elif len(idx_and_type) > 1:
                # multi-page CCRIS Details tables identified
                for item in idx_and_type:
                    lookup_table = self.__convert_to_row_map__(self.result.tables[item['idx']]) 
                    tagged_tables.append({
                        'type': item['type'],
                        'table': lookup_table
                    })
                    self.__record_table_pages__(item['idx'], item['type'])
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
        
        if self.report_type == ReportType.INDIVIDUAL:

            if self.__is_fuzzy_match__(header_text, 'credit info at a glance') or self.__is_fuzzy_match__(header_text, 'credit info') or self.__is_fuzzy_match__(header_text, 'bankruptcy proceedings record'):
                return [{'idx': table_idx, 'type': 'CREDIT_INFO_AT_A_GLANCE'}]
            
            if self.__is_fuzzy_match__(header_text, 'c1: banking payment records (source: ccris, bank negara malaysia)') or self.__is_fuzzy_match__(header_text, 'ccris entity key') or self.__is_fuzzy_match__(header_text, 'ccris summary') or self.__is_fuzzy_match__(header_text, 'credit applications') or self.__is_fuzzy_match__(header_text, 'approved in past 12 months') or self.__is_fuzzy_match__(header_text, 'summary of potential & current liabilities') or self.__is_fuzzy_match__(header_text, 'as borrower'):
                return [{'idx': table_idx, 'type': 'CCRIS_SUMMARY'}]
            
            if self.__is_fuzzy_match__(header_text, 'ccris details)') or self.__is_fuzzy_match__(header_text, 'loan information') or self.__is_fuzzy_match__(header_text, 'outstanding credit') or (self.__is_fuzzy_match__(header_text, 'no') and table.column_count == 25):
                return self.__handle_ccris_details_tables__(table_idx)
            
        elif self.report_type == ReportType.COMPANY:

            if self.__is_fuzzy_match__(header_text, 'a: snapshot') or self.__is_fuzzy_match__(header_text, 'id verification') or self.__is_fuzzy_match__(header_text, 'company name (your input)') or self.__is_fuzzy_match__(header_text, 'business name (your input)'):
                return [{'idx': table_idx, 'type': 'SNAPSHOT'}]
            
            if (self.__is_fuzzy_match__(header_text, 'financials and shareholders') or self.__is_fuzzy_match__(header_text, 'last updated')) and table.column_count == 2:
                return [{'idx': table_idx, 'type': 'FINANCIALS_AND_SHAREHOLDERS'}]
            
            if self.__is_fuzzy_match__(header_text, 'credit info at a glance') or self.__is_fuzzy_match__(header_text, 'credit info') or self.__is_fuzzy_match__(header_text, 'winding up / bankruptcy proceedings record'):
                return [{'idx': table_idx, 'type': 'CREDIT_INFO_AT_A_GLANCE'}]
            
            if (self.__is_fuzzy_match__(header_text, 'financial highlights') or self.__is_fuzzy_match__(header_text, 'financial year end') or self.__is_fuzzy_match__(header_text, 'date of tabling') or self.__is_fuzzy_match__(header_text, 'balance sheet') or self.__is_fuzzy_match__(header_text, 'non-current assets') or self.__is_fuzzy_match__(header_text, 'income statement') or self.__is_fuzzy_match__(header_text, 'revenue') or self.__is_fuzzy_match__(header_text, 'liquidity ratios') or self.__is_fuzzy_match__(header_text, 'current ratio')) and table.column_count == 6:
                return [{'idx': table_idx, 'type': 'FINANCIAL_STATEMENTS'}]
            
            if self.__is_fuzzy_match__(header_text, 'c1: banking payment records (source: ccris, bank negara malaysia)' or self.__is_fuzzy_match__(header_text, 'ccris entity key') or self.__is_fuzzy_match__(header_text, 'ccris summary') or self.__is_fuzzy_match__(header_text, 'credit applications') or self.__is_fuzzy_match__(header_text, 'approved in past 12 months') or self.__is_fuzzy_match__(header_text, 'summary of potential & current liabilities') or self.__is_fuzzy_match__(header_text, 'as borrower')):
                return [{'idx': table_idx, 'type': 'CCRIS_SUMMARY'}]
            
            if self.__is_fuzzy_match__(header_text, 'ccris details)') or self.__is_fuzzy_match__(header_text, 'loan information') or self.__is_fuzzy_match__(header_text, 'outstanding credit') or (self.__is_fuzzy_match__(header_text, 'no') and (table.column_count == 25 or table.column_count == 14)):
                return self.__handle_ccris_details_tables__(table_idx)

        # Second try: Use preceding paragraph
        if preceding_text:
            preceding_lower = preceding_text.strip().lower()

            if self.report_type == ReportType.INDIVIDUAL:  

                if self.__is_fuzzy_match__(preceding_lower, 'credit info at a glance') or self.__is_fuzzy_match__(preceding_lower, 'credit info') or self.__is_fuzzy_match__(preceding_lower, 'bankruptcy proceedings record'):
                    return [{'idx': table_idx, 'type': 'CREDIT_INFO_AT_A_GLANCE'}]
                
                if self.__is_fuzzy_match__(preceding_lower, 'c1: banking payment records (source: ccris, bank negara malaysia)') or self.__is_fuzzy_match__(preceding_lower, 'ccris entity key') or self.__is_fuzzy_match__(preceding_lower, 'ccris summary') or self.__is_fuzzy_match__(preceding_lower, 'credit applications') or self.__is_fuzzy_match__(preceding_lower, 'approved in past 12 months') or self.__is_fuzzy_match__(preceding_lower, 'summary of potential & current liabilities') or self.__is_fuzzy_match__(preceding_lower, 'as borrower'):
                    return [{'idx': table_idx, 'type': 'CCRIS_SUMMARY'}]
                
                if self.__is_fuzzy_match__(preceding_lower, 'ccris details)') or self.__is_fuzzy_match__(preceding_lower, 'loan information') or self.__is_fuzzy_match__(preceding_lower, 'outstanding credit') or (self.__is_fuzzy_match__(preceding_lower, 'no') and (table.column_count == 25 or table.column_count == 14)):
                    return self.__handle_ccris_details_tables__(table_idx)
                
            elif self.report_type == ReportType.COMPANY:

                if self.__is_fuzzy_match__(preceding_lower, 'a: snapshot') or self.__is_fuzzy_match__(preceding_lower, 'id verification') or self.__is_fuzzy_match__(preceding_lower, 'company name (your input)'):
                    return [{'idx': table_idx, 'type': 'SNAPSHOT'}]
                
                if (self.__is_fuzzy_match__(preceding_lower, 'financials and shareholders') or self.__is_fuzzy_match__(preceding_lower, 'last updated')) and table.column_count == 2:
                    return [{'idx': table_idx, 'type': 'FINANCIALS_AND_SHAREHOLDERS'}]
                
                if self.__is_fuzzy_match__(preceding_lower, 'credit info at a glance') or self.__is_fuzzy_match__(preceding_lower, 'credit info') or self.__is_fuzzy_match__(preceding_lower, 'winding up / bankruptcy proceedings record'):
                    return [{'idx': table_idx, 'type': 'CREDIT_INFO_AT_A_GLANCE'}]
                
                if (self.__is_fuzzy_match__(preceding_lower, 'financial highlights') or self.__is_fuzzy_match__(preceding_lower, 'financial year end') or self.__is_fuzzy_match__(preceding_lower, 'date of tabling') or self.__is_fuzzy_match__(preceding_lower, 'balance sheet') or self.__is_fuzzy_match__(preceding_lower, 'non-current assets') or self.__is_fuzzy_match__(preceding_lower, 'income statement') or self.__is_fuzzy_match__(preceding_lower, 'revenue') or self.__is_fuzzy_match__(preceding_lower, 'liquidity ratios') or self.__is_fuzzy_match__(preceding_lower, 'current ratio')) and table.column_count == 6:
                    return [{'idx': table_idx, 'type': 'FINANCIAL_STATEMENTS'}]
                
                if self.__is_fuzzy_match__(preceding_lower, 'c1: banking payment records (source: ccris, bank negara malaysia)') or self.__is_fuzzy_match__(preceding_lower, 'ccris entity key') or self.__is_fuzzy_match__(preceding_lower, 'ccris summary') or self.__is_fuzzy_match__(preceding_lower, 'credit applications') or self.__is_fuzzy_match__(preceding_lower, 'approved in past 12 months') or self.__is_fuzzy_match__(preceding_lower, 'summary of potential & current liabilities') or self.__is_fuzzy_match__(preceding_lower, 'as borrower'):
                    return [{'idx': table_idx, 'type': 'CCRIS_SUMMARY'}]
                
                if self.__is_fuzzy_match__(preceding_lower, 'ccris details)') or self.__is_fuzzy_match__(preceding_lower, 'loan information') or self.__is_fuzzy_match__(preceding_lower, 'outstanding credit') or (self.__is_fuzzy_match__(preceding_lower, 'no') and (table.column_count == 25 or table.column_count == 14)):
                    return self.__handle_ccris_details_tables__(table_idx)
                
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
            if (self.report_type == ReportType.INDIVIDUAL):
                values = self.__extract_from_table_individual__(tagged_table)
            elif (self.report_type == ReportType.COMPANY):
                values = self.__extract_from_table_company__(tagged_table)
            
            extracted.update(values)
        return extracted

    def __extract_from_table_individual__(self, tagged_table: Dict) -> Dict:
        """Extracts data from a single tagged table for individual report based on its type."""
        table_type = tagged_table['type']
        table = tagged_table['table']
        
        extracted_values = {}
        
        if table_type == 'CREDIT_INFO_AT_A_GLANCE':
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()

                # TODO check if bankruptcy necessary
                if self.__is_fuzzy_match__(row_key_text, 'bankruptcy proceedings record'):
                    val = self.__safe_get_cell__(table, r_idx, 2)
                    if val is not None:
                        extracted_values['bankruptcy'] = val

                if self.__is_fuzzy_match__(row_key_text, 'legal records in past 24 months (non-personal capacity)'):
                    val = self.__safe_get_cell__(table, r_idx, 2)
                    if val is not None:
                        extracted_values['legal_non_personal'] = val
                
                if self.__is_fuzzy_match__(row_key_text, 'legal records in past 24 months (personal capacity)'):
                    val = self.__safe_get_cell__(table, r_idx, 2)
                    if val is not None:
                        extracted_values['legal_personal'] = val

                if self.__is_fuzzy_match__(row_key_text, 'special attention accounts'):
                    val = self.__safe_get_cell__(table, r_idx, 2)
                    if val is not None:
                        extracted_values['special_attention_accounts_0'] = val
                  
        elif table_type == 'CCRIS_SUMMARY':
            ccris_summary = self.__extract_from_ccris_summary__(table)
            extracted_values.update(ccris_summary)
        
        elif table_type == 'CCRIS_DETAILS_SINGLE':
            ccris_details = self.__extract_from_ccris_details_single__(table)
            extracted_values.update(ccris_details)

        elif table_type == 'CCRIS_DETAILS_MULTI_START':
            ccris_details = self.__extract_from_ccris_details_multi_start__(table)
            extracted_values.update(ccris_details)
        
        elif table_type == 'CCRIS_DETAILS_MULTI_MID':
            ccris_details = self.__extract_from_ccris_details_multi_mid__(table)
            extracted_values.update(ccris_details)

        elif table_type == 'CCRIS_DETAILS_MULTI_END':
            ccris_details = self.__extract_from_ccris_details_multi_end__(table)
            extracted_values.update(ccris_details)
             
        return extracted_values

    def __extract_from_ccris_summary__(self, table) -> Dict:
        """Extracts data from CCRIS Summary table."""
        extracted_values = {}
        for r_idx in table:
            row_key_text = table[r_idx].get(0, "").strip().lower()
            if self.__is_fuzzy_match__(row_key_text, 'as borrower'):
                val1 = self.__safe_get_cell__(table, r_idx, 1)
                val2 = self.__safe_get_cell__(table, r_idx, 2)
                if val1 is not None:
                    extracted_values['total_outstanding_balance_0'] = val1
                if val2 is not None:
                    extracted_values['total_limit_0'] = val2
            
            if self.__is_fuzzy_match__(row_key_text, 'special attention account'):
                val = self.__safe_get_cell__(table, r_idx, 1)
                if val is not None:
                    extracted_values['special_attention_accounts_1'] = val
        return extracted_values 
    
    def __extract_from_ccris_details_single__(self, table) -> Dict:
        """Extracts data from CCRIS Details Single table."""
        extracted_values = {}
        end_row_idx = -1
        start_row_idx = -1
        for r_idx in table:
            row_key_text = table[r_idx].get(5, "").strip().lower()
            if self.__is_fuzzy_match__(row_key_text, 'total outstanding balance'):
                val6 = self.__safe_get_cell__(table, r_idx, 6)
                val8 = self.__safe_get_cell__(table, r_idx, 8)
                if val6 is not None:
                    extracted_values['total_outstanding_balance_1'] = val6
                if val8 is not None:
                    extracted_values['total_limit_1'] = val8
                end_row_idx = r_idx

        for r_idx in table:
            header_text = table[r_idx].get(0, "").strip().lower()
            if self.__is_fuzzy_match__(header_text, 'outstanding credit'):
                start_row_idx = r_idx + 1

        if (start_row_idx != -1 and end_row_idx != -1):
            conduct_values = self.__extract_conduct__(table, start_row_idx, end_row_idx)
            extracted_values['ccris_conduct'] = conduct_values
        return extracted_values

    def __extract_from_ccris_details_multi_start__(self, table) -> Dict:
        """Extracts data from CCRIS Details Multi Start table."""
        extracted_values = {}
        start_row_idx = -1
        for r_idx in table:
            header_text = table[r_idx].get(0, "").strip().lower()
            if self.__is_fuzzy_match__(header_text, 'outstanding credit'):
                start_row_idx = r_idx + 1

        if (start_row_idx != -1):
            conduct_values = self.__extract_conduct__(table, start_row_idx, len(table))
            extracted_values['ccris_conduct'] = conduct_values
        return extracted_values
    
    def __extract_from_ccris_details_multi_mid__(self, table) -> Dict:
        """Extracts data from CCRIS Details Multi Mid table."""
        extracted_values = {}
        conduct_values = self.__extract_conduct__(table, 0, len(table))
        extracted_values['ccris_conduct'] = conduct_values
        return extracted_values
    
    def __extract_from_ccris_details_multi_end__(self, table) -> Dict:
        """Extracts data from CCRIS Details Multi End table."""
        extracted_values = {}
        end_row_idx = -1
        for r_idx in table:
            row_key_text = table[r_idx].get(5, "").strip().lower()
            if self.__is_fuzzy_match__(row_key_text, 'total outstanding balance'):
                val6 = self.__safe_get_cell__(table, r_idx, 6)
                val8 = self.__safe_get_cell__(table, r_idx, 8)
                if val6 is not None:
                    extracted_values['total_outstanding_balance_1'] = val6
                if val8 is not None:
                    extracted_values['total_limit_1'] = val8
                end_row_idx = r_idx

        if (end_row_idx != -1):
            conduct_values = self.__extract_conduct__(table, 0, end_row_idx)
            extracted_values['ccris_conduct'] = conduct_values
        return extracted_values
    
    def __extract_from_table_company__(self, tagged_table: Dict) -> Dict:
        """Extracts data from a single tagged table for company report based on its type."""
        table_type = tagged_table['type']
        table = tagged_table['table']
        
        extracted_values = {}
        
        if table_type == 'CREDIT_INFO_AT_A_GLANCE':
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()

                # TODO flag for human review
                if self.__is_fuzzy_match__(row_key_text, 'winding up / bankruptcy proceedings record'):
                    val2 = self.__safe_get_cell__(table, r_idx, 2)
                    val3 = self.__safe_get_cell__(table, r_idx, 3)
                    if val2 is not None:
                        extracted_values['bankruptcy_entity'] = val2
                    if val3 is not None:
                        extracted_values['bankruptcy_rp'] = val3

                if self.__is_fuzzy_match__(row_key_text, 'legal records in past 24 months (non-personal capacity)'):
                    val2 = self.__safe_get_cell__(table, r_idx, 2)
                    val3 = self.__safe_get_cell__(table, r_idx, 3)
                    if val2 is not None:
                        extracted_values['legal_non_personal_entity'] = val2
                    if val3 is not None:
                        extracted_values['legal_non_personal_rp'] = val3
                
                if self.__is_fuzzy_match__(row_key_text, 'legal records in past 24 months (personal capacity)'):
                    val2 = self.__safe_get_cell__(table, r_idx, 2)
                    val3 = self.__safe_get_cell__(table, r_idx, 3)
                    if val2 is not None:
                        extracted_values['legal_personal_entity'] = val2
                    if val3 is not None:
                        extracted_values['legal_personal_rp'] = val3
                   
                if self.__is_fuzzy_match__(row_key_text, 'special attention accounts'):
                    val2 = self.__safe_get_cell__(table, r_idx, 2)
                    val3 = self.__safe_get_cell__(table, r_idx, 3)
                    if val2 is not None:
                        extracted_values['special_attention_accounts_entity'] = val2
                    if val3 is not None:
                        extracted_values['special_attention_accounts_rp'] = val3
                  
        elif table_type == 'CCRIS_SUMMARY':
            ccris_summary = self.__extract_from_ccris_summary__(table)
            extracted_values.update(ccris_summary)
        
        elif table_type == 'CCRIS_DETAILS_SINGLE':
            ccris_details = self.__extract_from_ccris_details_single__(table)
            extracted_values.update(ccris_details)

        elif table_type == 'CCRIS_DETAILS_MULTI_START':
            ccris_details = self.__extract_from_ccris_details_multi_start__(table)
            extracted_values.update(ccris_details)
        
        elif table_type == 'CCRIS_DETAILS_MULTI_MID':
            ccris_details = self.__extract_from_ccris_details_multi_mid__(table)
            extracted_values.update(ccris_details)

        elif table_type == 'CCRIS_DETAILS_MULTI_END':
            ccris_details = self.__extract_from_ccris_details_multi_end__(table)
            extracted_values.update(ccris_details)
       
        elif table_type == 'SNAPSHOT':
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()
                if self.__is_fuzzy_match__(row_key_text, 'registration date'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['registration_date'] = val
                
                if self.__is_fuzzy_match__(row_key_text, 'type', 100):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['type'] = " ".join(val.splitlines())
                
                if self.__is_fuzzy_match__(row_key_text, 'type of company'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['type'] = " ".join(val.splitlines())

                if self.__is_fuzzy_match__(row_key_text, 'msic'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['msic'] = " ".join(val.splitlines())

                if self.__is_fuzzy_match__(row_key_text, 'business commenced') or self.__is_fuzzy_match__(row_key_text, 'last changed date') or self.__is_fuzzy_match__(row_key_text, 'rob search date') or self.__is_fuzzy_match__(row_key_text, 'current registration expiry date'):
                    self.relevant_paras['partnership'] = True
        
        elif table_type == 'FINANCIALS_AND_SHAREHOLDERS':
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()
                if self.__is_fuzzy_match__(row_key_text, 'revenue (rm)'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['revenue_0'] = val
                if self.__is_fuzzy_match__(row_key_text, 'profit after tax (rm)'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['profit_after_tax_0'] = val
                if self.__is_fuzzy_match__(row_key_text, 'paid up capital (rm)'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['paid_up_capital'] = val

        elif table_type == 'FINANCIAL_STATEMENTS':
            for r_idx in table:
                row_key_text = table[r_idx].get(0, "").strip().lower()
                if self.__is_fuzzy_match__(row_key_text, 'financial year end'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['financial_year_end'] = val
                
                if self.__is_fuzzy_match__(row_key_text, 'non-current assets', 95):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['non_current_assets'] = val
                    
                if self.__is_fuzzy_match__(row_key_text, 'current assets', 95):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['current_assets'] = val
                    
                if self.__is_fuzzy_match__(row_key_text, 'total assets', 95):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['total_assets'] = val

                if self.__is_fuzzy_match__(row_key_text, 'non-current liabilities', 95):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['non_current_liabilities'] = val
                
                if self.__is_fuzzy_match__(row_key_text, 'current liabilities', 95):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['current_liabilities'] = val
                
                if self.__is_fuzzy_match__(row_key_text, 'long term liabilities', 95):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['long_term_liabilities'] = val
                
                if self.__is_fuzzy_match__(row_key_text, 'total liabilities', 95):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['total_liabilities'] = val
                     
                if self.__is_fuzzy_match__(row_key_text, 'retained earning'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['retained_earning'] = val

                if self.__is_fuzzy_match__(row_key_text, 'net worth (ta - tl)'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['net_worth'] = val
                
                if self.__is_fuzzy_match__(row_key_text, 'revenue'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['revenue_1'] = val

                if self.__is_fuzzy_match__(row_key_text, 'profit / (loss) after tax'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['profit_after_tax_1'] = val

                if self.__is_fuzzy_match__(row_key_text, 'current ratio'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['current_ratio'] = val

                if self.__is_fuzzy_match__(row_key_text, 'gearing ratio'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['gearing_ratio'] = val

                if self.__is_fuzzy_match__(row_key_text, 'debt to equity ratio [%]'):
                    val = self.__safe_get_cell__(table, r_idx, 1)
                    if val is not None:
                        extracted_values['debt_to_equity_ratio'] = val

        return extracted_values

    def __parse_extracted_data__(self, extracted_data: Dict):
        """Parses and validates extracted data into final structured format."""
        logger.debug("Parsing extracted data keys: %s", list(extracted_data.keys()))

        parsed_data = {}
        if self.report_type == ReportType.INDIVIDUAL:
               
            if extracted_data.get('ccris_conduct') is not None and extracted_data.get('total_outstanding_balance_1') is None and extracted_data.get('total_limit_1') is None:
                logger.info("CCRIS Detail edge case detected.")
                details_image_data = self.extract_using_image('ccris_detail_edge_case')
            else:
                details_image_data = self.extract_using_image('ccris_detail')

            if details_image_data:
                if details_image_data.get('ccris_conduct') is not None:
                    logger.info("CCRIS Conduct data extracted from image: %s", details_image_data['ccris_conduct'])
                    parsed_data['repayment_to_banks'] = self.__parse_conduct_values_image__(details_image_data['ccris_conduct'])
                
                if extracted_data.get('total_outstanding_balance_1') is None and details_image_data.get('total_outstanding_balance_1') is not None:
                    extracted_data['total_outstanding_balance_1'] = details_image_data['total_outstanding_balance_1']
                
                if extracted_data.get('total_limit_1') is None and details_image_data.get('total_limit_1') is not None:
                    extracted_data['total_limit_1'] = details_image_data['total_limit_1']
            else:
                logger.error("CCRIS Details image extraction failed. Using fallback from document intelligence. This may impact the accuracy of these fields.")
            
            # parsed ccris_conduct from document intelligence is only used as a reference. The actual repayment_to_banks value is extracted from image extraction.
            if parsed_data.get('repayment_to_banks') is None:
                if extracted_data.get('ccris_conduct'):
                    parsed_data['repayment_to_banks'] = self.__parse_conduct_values__(extracted_data['ccris_conduct'])
                else:
                    logger.error("CCRIS Conduct data not found in both table and image extraction. Repayment to banks field will be missing.")

            util_keys = ['total_outstanding_balance_0', 'total_outstanding_balance_1', 'total_limit_0', 'total_limit_1']
            if all(extracted_data.get(key) is not None for key in util_keys):
                if self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_0']) == self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_1']) and self.__normalize_numeric_str__(extracted_data['total_limit_0']) == self.__normalize_numeric_str__(extracted_data['total_limit_1']):
                    bal = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_0']))
                    limit = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['total_limit_0']))
                    if limit > 0:
                        utilisation = bal / limit * 100
                        parsed_data['utilisation'] = utilisation

            spa_keys = ['special_attention_accounts_0', 'special_attention_accounts_1']
            if all(extracted_data.get(key) is not None for key in spa_keys):
                # saa_0 is 'NO', saa_1 is 'N'
                str = extracted_data['special_attention_accounts_0']
                if str[0] == extracted_data['special_attention_accounts_1']:
                    parsed_data['special_attention_accounts'] = extracted_data['special_attention_accounts_0']

            legal_keys = ['legal_non_personal', 'legal_personal']
            if all(extracted_data.get(key) is not None for key in legal_keys):
                if extracted_data['legal_non_personal'] == '0' and extracted_data['legal_personal'] == '0':
                    parsed_data['legal_cases'] = 0
                else:
                    np = int(extracted_data['legal_non_personal'])
                    p = int(extracted_data['legal_personal'])
                    parsed_data['legal_cases'] = np + p

            # TODO check blacklist

            # use ai as fallback. this needs to be async
            if parsed_data.get('utilisation') is None:
                logger.info("Utilisation not found from table extraction, falling back to image extraction")
                summary_image_data = self.extract_using_image('ccris_summary')
                if summary_image_data:
                    if summary_image_data.get('total_outstanding_balance') is not None and summary_image_data.get('total_limit') is not None:
                        bal = self.__str_to_decimal__(self.__normalize_numeric_str__(summary_image_data['total_outstanding_balance']))
                        limit = self.__str_to_decimal__(self.__normalize_numeric_str__(summary_image_data['total_limit']))
                        if limit > 0:
                            utilisation = bal / limit * 100
                            parsed_data['utilisation'] = utilisation
                    
                    if parsed_data.get('special_attention_accounts') is None and summary_image_data.get('special_attention_accounts') is not None:
                        parsed_data['special_attention_accounts'] = summary_image_data['special_attention_accounts']

            if (parsed_data.get('special_attention_accounts') is None) or (parsed_data.get('legal_cases') is None):
                logger.info("Special attention accounts or legal cases not found from table extraction, falling back to image extraction")
                credit_image_data = self.extract_using_image('credit_info_at_a_glance')
                if credit_image_data:
                    if parsed_data.get('special_attention_accounts') is None and credit_image_data.get('special_attention_accounts') is not None:
                        parsed_data['special_attention_accounts'] = credit_image_data['special_attention_accounts']
                    else:
                        logger.error("Special attention accounts not found in image extraction for special_attention_accounts assignment")

                    if parsed_data.get('legal_cases') is None and credit_image_data.get('legal_non_personal') is not None and credit_image_data.get('legal_personal') is not None:
                        if credit_image_data['legal_non_personal'] == '0' and credit_image_data['legal_personal'] == '0':
                            parsed_data['legal_cases'] = 0
                        else:
                            np = int(credit_image_data['legal_non_personal'])
                            p = int(credit_image_data['legal_personal'])
                            parsed_data['legal_cases'] = np + p
                    else:
                        logger.error("Legal cases not found in image extraction for legal_cases assignment")


            
        elif self.report_type == ReportType.COMPANY:
            
            util_keys = ['total_outstanding_balance_0', 'total_outstanding_balance_1', 'total_limit_0', 'total_limit_1']
            if all(extracted_data.get(key) is None for key in util_keys) and extracted_data.get('ccris_conduct') is None:
                parsed_data['repayment_to_banks'] = 'N/A'
                parsed_data['utilisation'] = 'N/A'
            else:
                if extracted_data.get('ccris_conduct') is not None and extracted_data.get('total_outstanding_balance_1') is None and extracted_data.get('total_limit_1') is None:
                    details_image_data = self.extract_using_image('ccris_detail_edge_case')
                else:
                    details_image_data = self.extract_using_image('ccris_detail')

                if details_image_data:
                    if details_image_data.get('ccris_conduct') is not None:
                        logger.info("CCRIS Conduct data extracted from image: %s", details_image_data['ccris_conduct'])
                        parsed_data['repayment_to_banks'] = self.__parse_conduct_values_image__(details_image_data['ccris_conduct'])
                    
                    if extracted_data.get('total_outstanding_balance_1') is None and details_image_data.get('total_outstanding_balance_1') is not None:
                        extracted_data['total_outstanding_balance_1'] = details_image_data['total_outstanding_balance_1']
                    
                    if extracted_data.get('total_limit_1') is None and details_image_data.get('total_limit_1') is not None:
                        extracted_data['total_limit_1'] = details_image_data['total_limit_1']
                else:
                    logger.error("CCRIS Details image extraction failed. Using fallback from document intelligence. This may impact the accuracy of these fields.")
                
                # parsed ccris_conduct from document intelligence is only used as a reference. The actual repayment_to_banks value is extracted from image extraction.
                if parsed_data.get('repayment_to_banks') is None:
                    if extracted_data.get('ccris_conduct'):
                        parsed_data['repayment_to_banks'] = self.__parse_conduct_values__(extracted_data['ccris_conduct'])
                    else:
                        logger.error("CCRIS Conduct data not found in both table and image extraction. Repayment to banks field will be missing.")
                
                util_keys = ['total_outstanding_balance_0', 'total_outstanding_balance_1', 'total_limit_0', 'total_limit_1']
                if all(extracted_data.get(key) is not None for key in util_keys):
                    if self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_0']) == self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_1']) and self.__normalize_numeric_str__(extracted_data['total_limit_0']) == self.__normalize_numeric_str__(extracted_data['total_limit_1']):
                        bal = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['total_outstanding_balance_0']))
                        limit = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['total_limit_0']))
                        if limit > 0:
                            utilisation = bal / limit * 100
                            parsed_data['utilisation'] = utilisation

                if parsed_data.get('utilisation') is None:
                    logger.info("Utilisation not found from table extraction, falling back to image extraction")
                    summary_image_data = self.extract_using_image('ccris_summary')
                    if summary_image_data and summary_image_data.get('total_outstanding_balance') is not None and summary_image_data.get('total_limit') is not None:
                        bal = self.__str_to_decimal__(self.__normalize_numeric_str__(summary_image_data['total_outstanding_balance']))
                        limit = self.__str_to_decimal__(self.__normalize_numeric_str__(summary_image_data['total_limit']))
                        if limit > 0:
                            utilisation = bal / limit * 100
                            parsed_data['utilisation'] = utilisation
                    else:
                        logger.error("Utilisation data not found in image extraction for utilisation calculation")
                        

            if extracted_data.get('special_attention_accounts_entity') is not None:
                parsed_data['special_attention_accounts'] = extracted_data['special_attention_accounts_entity']

            if extracted_data.get('legal_non_personal_entity') is not None and extracted_data.get('legal_personal_entity') is not None:
                if extracted_data['legal_non_personal_entity'] == '0' and extracted_data['legal_personal_entity'] == '0':
                    parsed_data['legal_cases'] = 0
                else:
                    np = int(extracted_data['legal_non_personal_entity'])
                    p = int(extracted_data['legal_personal_entity'])
                    parsed_data['legal_cases'] = np + p

            if (parsed_data.get('special_attention_accounts') is None) or (parsed_data.get('legal_cases') is None):
                logger.info("Special attention accounts or legal cases not found from table extraction, falling back to image extraction")
                credit_image_data = self.extract_using_image('credit_info_at_a_glance')
                if credit_image_data:
                    if parsed_data.get('special_attention_accounts') is None and credit_image_data.get('special_attention_accounts_entity') is not None:
                        parsed_data['special_attention_accounts'] = credit_image_data['special_attention_accounts_entity']
                    else:
                        logger.error("Special attention accounts not found in image extraction for special_attention_accounts assignment")

                    if parsed_data.get('legal_cases') is None and credit_image_data.get('legal_non_personal_entity') is not None and credit_image_data.get('legal_personal_entity') is not None:
                        if credit_image_data['legal_non_personal_entity'] == '0' and credit_image_data['legal_personal_entity'] == '0':
                            parsed_data['legal_cases'] = 0
                        else:
                            np = int(credit_image_data['legal_non_personal_entity'])
                            p = int(credit_image_data['legal_personal_entity'])
                            parsed_data['legal_cases'] = np + p
                    else:
                        logger.error("Legal cases not found in image extraction for legal_cases assignment")

            # DEBUG. CHECK WHICH KEYS ARE NOT PRESENT
            for key in ['special_attention_accounts', 'legal_cases', 'utilisation', 'repayment_to_banks']:
                if parsed_data.get(key) is None:
                    logger.error("Key %s not found in parsed data", key)

            # TODO check blacklist

            if extracted_data.get('registration_date') is not None:
                parsed_data['years_in_business'] = self.__calculate_years__(extracted_data['registration_date'])

            if extracted_data.get('type') is not None:
                type = extracted_data['type']
                if self.__is_fuzzy_match__(type, 'limited by shares private limited'):
                    parsed_data['type_of_company'] = 'Sdn Bhd'
                else: 
                    parsed_data['type_of_company'] = 'Non - Sdn Bhd'
            
            if extracted_data.get('msic') is not None:
                parsed_data['nature_of_business'] = extracted_data['msic']
            
            if parsed_data.get('years_in_business') is None or parsed_data.get('type_of_company') is None or parsed_data.get('nature_of_business') is None:
                logger.info("Snapshot data incomplete from table extraction, falling back to image extraction")
                snapshot_image_data = self.extract_using_image('snapshot')
                if snapshot_image_data:
                    if parsed_data.get('years_in_business') is None and snapshot_image_data.get('registration_date') is not None:
                        parsed_data['years_in_business'] = self.__calculate_years__(snapshot_image_data['registration_date'])
                    
                    if parsed_data.get('type_of_company') is None and snapshot_image_data.get('type') is not None:
                        parsed_data['type_of_company'] = snapshot_image_data['type']
                    
                    if parsed_data.get('nature_of_business') is None and snapshot_image_data.get('msic') is not None:
                        parsed_data['nature_of_business'] = snapshot_image_data['msic']
                    
                    if snapshot_image_data.get('is_partnership') == True:
                        self.relevant_paras['partnership'] = True

            # DEBUG. CHECK WHICH KEYS ARE NOT PRESENT
            for key in ['years_in_business', 'type_of_company', 'nature_of_business']:
                if parsed_data.get(key) is None:
                    logger.error("Key %s not found in parsed data", key)

            if self.relevant_paras.get('partnership') is not None and parsed_data.get('type_of_company') == 'Non - Sdn Bhd':
                parsed_data['paid_up_capital'] = 'N/A'
                parsed_data['financial_report_date'] = 'N/A'
                parsed_data['turnover'] = 'N/A'
                parsed_data['net_profit'] = 'N/A'
                parsed_data['retained_profit'] = 'N/A'
                parsed_data['net_worth'] = 'N/A'
                parsed_data['net_current_assets'] = 'N/A'
                parsed_data['current_ratio'] = 'N/A'
                parsed_data['gearing_ratio'] = 'N/A'
            else:
                if extracted_data.get('paid_up_capital') is not None:
                    parsed_data['paid_up_capital'] = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['paid_up_capital']))
            
                if extracted_data.get('financial_year_end') is not None:
                    parsed_data['financial_report_date'] = self.__reformat_date__(extracted_data['financial_year_end'])

                if extracted_data.get('revenue_0') is not None and extracted_data.get('revenue_1') is not None:
                    if self.__normalize_numeric_str__(extracted_data['revenue_0']) == self.__normalize_numeric_str__(extracted_data['revenue_1']):
                        parsed_data['turnover'] = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['revenue_0']))
                    elif self.__normalize_numeric_str__(extracted_data['revenue_0']) == '0' and self.__normalize_numeric_str__(extracted_data['revenue_1']) == '0':
                        parsed_data['turnover'] = Decimal(0)
                    
                if extracted_data.get('profit_after_tax_0') is not None and extracted_data.get('profit_after_tax_1') is not None:
                    if self.__normalize_numeric_str__(extracted_data['profit_after_tax_0']) == self.__normalize_numeric_str__(extracted_data['profit_after_tax_1']):
                        parsed_data['net_profit'] = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['profit_after_tax_0']))
                    elif self.__normalize_numeric_str__(extracted_data['profit_after_tax_0']) == '0' and self.__normalize_numeric_str__(extracted_data['profit_after_tax_1']) == '0':
                        parsed_data['net_profit'] = Decimal(0)
                 
                if extracted_data.get('retained_earning') is not None:
                    parsed_data['retained_profit'] = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['retained_earning']))

                if extracted_data.get('net_worth') is not None:
                    parsed_data['net_worth'] = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['net_worth']))

                fs_key = ['current_assets', 'current_liabilities', 'non_current_assets', 'total_assets', 'non_current_liabilities', 'long_term_liabilities', 'total_liabilities']
                if all(extracted_data.get(key) is not None for key in fs_key):
                    nca = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['non_current_assets']))
                    ca = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['current_assets']))
                    ta = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['total_assets']))
                    ncl = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['non_current_liabilities']))
                    cl = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['current_liabilities']))
                    ltl = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['long_term_liabilities']))
                    tl = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['total_liabilities']))

                    valid_ca_cl = (ta == nca + ca) and (tl == ncl + cl + ltl)
                    if valid_ca_cl:
                        parsed_data['net_current_assets'] = ca - cl
                        if extracted_data.get('current_ratio') is not None:
                            extracted_cr = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['current_ratio']))
                            cr = ca / cl if cl > 0 else Decimal(0)
                            if abs(cr - extracted_cr) < Decimal('0.01'):
                                parsed_data['current_ratio'] = extracted_cr

                bal_key = ['gearing_ratio', 'debt_to_equity_ratio', 'net_worth', 'total_liabilities']
                if all(extracted_data.get(key) is not None for key in bal_key):
                    tl = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['total_liabilities']))
                    nw = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['net_worth']))
                    calculated_gr = tl / nw if nw > 0 else Decimal(0)
                    extracted_gr = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['gearing_ratio']))
                    extracted_der = self.__str_to_decimal__(self.__normalize_numeric_str__(extracted_data['debt_to_equity_ratio']))
                    valid_gr = (abs(extracted_gr - extracted_der) < Decimal('0.01')) and (abs(extracted_gr - calculated_gr) < Decimal('0.01'))
                    if valid_gr:
                        parsed_data['gearing_ratio'] = extracted_gr
                
                # DEBUG. CHECK WHICH KEYS ARE NOT PRESENT
                for key in ['paid_up_capital', 'financial_report_date', 'turnover', 'net_profit', 'retained_profit', 'net_worth', 'net_current_assets', 'current_ratio', 'gearing_ratio']:
                    if parsed_data.get(key) is None:
                        logger.error("Key %s not found in parsed data", key)

                if parsed_data.get('paid_up_capital') is None:
                    logger.info("Paid up capital not found from table extraction, falling back to image extraction")
                    shareholders_image_data = self.extract_using_image('financials_and_shareholders')
                    if shareholders_image_data and shareholders_image_data.get('paid_up_capital') is not None:
                        parsed_data['paid_up_capital'] = self.__str_to_decimal__(self.__normalize_numeric_str__(shareholders_image_data['paid_up_capital']))
                    else:
                        logger.error("Paid up capital not found in image extraction")

                if (parsed_data.get('financial_report_date') is None) or (parsed_data.get('turnover') is None) or (parsed_data.get('net_profit') is None) or (parsed_data.get('retained_profit') is None) or (parsed_data.get('net_worth') is None) or (parsed_data.get('net_current_assets') is None) or (parsed_data.get('current_ratio') is None) or (parsed_data.get('gearing_ratio') is None):
                    logger.info("Financial statements data incomplete from table extraction, falling back to image extraction")
                    financials_image_data = self.extract_using_image('financial_statements')
                    logger.info("Financial statements data extracted from image: %s", json.dumps(financials_image_data, indent=2))

                    if financials_image_data:
                        if parsed_data.get('financial_report_date') is None and financials_image_data.get('financial_year_end') is not None:
                            parsed_data['financial_report_date'] = financials_image_data['financial_year_end']
                        

                        if parsed_data.get('turnover') is None and financials_image_data.get('revenue') is not None:
                            parsed_data['turnover'] = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['revenue']))
                        
                        
                        if parsed_data.get('net_profit') is None and financials_image_data.get('profit_after_tax') is not None:
                            parsed_data['net_profit'] = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['profit_after_tax']))
                        
                        
                        if parsed_data.get('retained_profit') is None and financials_image_data.get('retained_earning') is not None:
                            parsed_data['retained_profit'] = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['retained_earning']))
                        
                        
                        if parsed_data.get('net_worth') is None and financials_image_data.get('net_worth') is not None:
                            parsed_data['net_worth'] = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['net_worth']))
                        
                        
                        if parsed_data.get('net_current_assets') is None:
                            fs_key = ['current_assets', 'current_liabilities', 'non_current_assets', 'total_assets', 'non_current_liabilities', 'long_term_liabilities', 'total_liabilities']
                            if all(financials_image_data.get(key) is not None for key in fs_key):
                                nca = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['non_current_assets']))
                                ca = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['current_assets']))
                                ta = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['total_assets']))
                                ncl = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['non_current_liabilities']))
                                cl = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['current_liabilities']))
                                ltl = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['long_term_liabilities']))
                                tl = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['total_liabilities']))
                                valid_ca_cl = (ta == nca + ca) and (tl == ncl + cl + ltl)
                                if valid_ca_cl:
                                    parsed_data['net_current_assets'] = ca - cl
                                else:
                                    logger.error("Current assets and liabilities validation failed in image extraction")
                        
                        if parsed_data.get('current_ratio') is None and financials_image_data.get('current_ratio') is not None:
                            extracted_cr = self.__str_to_decimal__(financials_image_data['current_ratio'])
                            ca = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['current_assets']))
                            cl = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['current_liabilities']))
                            cr = ca / cl if cl > 0 else Decimal(0)
                            if abs(cr - extracted_cr) < Decimal('0.01'):
                                parsed_data['current_ratio'] = extracted_cr
                            else:
                                logger.error("Current ratio validation failed in image extraction")
                        
                        if parsed_data.get('gearing_ratio') is None and financials_image_data.get('gearing_ratio') is not None and financials_image_data.get('net_worth') is not None and financials_image_data.get('total_liabilities') is not None:
                            tl = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['total_liabilities']))
                            nw = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['net_worth']))
                            calculated_gr = tl / nw if nw > 0 else Decimal(0)
                            extracted_gr = self.__str_to_decimal__(self.__normalize_numeric_str__(financials_image_data['gearing_ratio']))
                            valid_gr = (abs(extracted_gr - calculated_gr) < Decimal('0.01'))
                            if valid_gr:
                                parsed_data['gearing_ratio'] = extracted_gr
                            else:
                                logger.error("Gearing ratio validation failed in image extraction")

        return parsed_data   
    
    def extract_using_image(self, table_tag: str):
        """Extract data from images of document pages where the specified table is located."""
        client = self.__get_openai_client__(self.options)

        page_start, page_end = self.__get_page_range_for_table_tag__(table_tag)

        if page_start is None or page_end is None:
            page_start, page_end = 1, len(self.result.pages)

        image_uris = self.__get_document_image_uris__(
            self.bytes, page_start, page_end)
        
        table_prompt = self.__get_prompt_for_table_tag__(table_tag)

        user_content = [{"type": "text", "text": table_prompt}]

        for image_uri in image_uris:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": image_uri,
                    "detail": "high"
                }
            })

        # 1. Change to .create() instead of .beta...parse()
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
            # 2. Set response_format to json_object
            response_format={"type": "json_object"}
        )

        # 3. Access .content (a string) and parse it manually
        raw_content = completion.choices[0].message.content
        
        try:
            response_obj_dict = json.loads(raw_content)
            return response_obj_dict
        except json.JSONDecodeError:
            # Fallback in case the model returns invalid JSON 
            # (Rare with json_object mode, but good practice)
            return {"error": "Failed to decode JSON", "raw": raw_content}
        
        '''
        completion = client.beta.chat.completions.parse(
            model=self.options.deployment_name,
            messages=[
                {
                    "role": "system",
                    "content": self.options.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            max_tokens=self.options.max_tokens,
            temperature=self.options.temperature,
            top_p=self.options.top_p,
            # Enabled to determine the confidence of the response.
            logprobs=True
        )

        response_obj = completion.choices[0].message.parsed
        response_obj_dict = response_obj.model_dump()
        return response_obj_dict
        '''

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
        }
        
        types = tag_to_types.get(table_tag, [])
        min_page, max_page = None, None
        
        for t in types:
            if t in self.table_page_ranges:
                t_min, t_max = self.table_page_ranges[t]
                min_page = t_min if min_page is None else min(min_page, t_min)
                max_page = t_max if max_page is None else max(max_page, t_max)
        
        # Edge case: CCRIS_DETAILS_SINGLE may actually be a multi-page table where the number of columns parsed for the first part of the table and subsequent parts are different. Since __detect_ccris_details_tables__ relies on column_count, it gets tagged as SINGLE. A future fix would be to detect the boilerplate between multi-page tables, or parsing the markdown, for a more robust detect method. To handle this case, we extend the end page to the end of the document so the LLM can find the rest of the table.
        if table_tag == 'ccris_detail_edge_case':
            max_page = len(self.result.pages)

        return (min_page, max_page)
            
    def __get_prompt_for_table_tag__(self, table_tag: str) -> str:
        """Returns the prompt string for a given table tag."""
        match table_tag:
            case 'ccris_summary':
                return (
                    "Extract the following fields from the table with the heading 'C1: BANKING PAYMENT RECORDS (SOURCE: CCRIS, BANK NEGARA MALAYSIA)'. Under the subheading 'Summary of Potential & Current Liabilities', for the first row labeled 'As Borrower', extract the two values of total outstanding balance and total limit from the columns 'Outstanding' and 'Total Limit'. Do not confuse this with the second row labeled 'As Guarantor'. Do not confuse this with the third row labeled 'Total'. If the value is 0, it may be represented as a dash '-' or an en-dash '–' or an em-dash '—'. If the value is 0.00, return 0.00 and do not return null. Brackets surrounding a numerical value indicates that the numerical value is negative. Extract the value ('Y' or 'N') for the field 'Special Attention Account' which is the last row of the table, under the column 'Outstanding'. If any of these fields are not present in the table, return null for that field. Return the extracted data in the following JSON format: {\"total_outstanding_balance\": value or null, \"total_limit\": value or null, \"special_attention_accounts\": value or null}."
                )
            case 'ccris_detail':
                return (
                    "Attached are images of pages from a credit report containing a table with the heading 'CCRIS Details' and subheadings 'Loan Information', 'Special Attention Account', and 'Credit Application'. The columns are: 'No', 'Date', 'Sts', 'Capacity', 'Lender Type', 'Facility', 'Total Outstanding Balance', 'Data Balance Updated', 'Limit/Installment Amount', 'Prin. Repmt. Term', 'Col Type', 'Conduct of Account For Last 12 Months', 'LGL STS', and 'Date Status Updated'. The column 'Conduct of Account For Last 12 Months' contains 12 sub-columns representing the repayment conduct for each of the last 12 months, with values representing the number of months the payment was late (0 for on-time payment). We are only interested in extracting data from the column 'Conduct of Account For Last 12 Months' and the summary row showing 'Total Outstanding Balance' and 'Total Limit' right before the subheading 'Special Attention Account'. "
                    "The 'CCRIS Details' table may span multiple pages. "
                    "The 'CCRIS Details' table ends when you encounter the 'Remark Legend', or any section header that is clearly not part of the CCRIS details table. "
                    "Extract the following from the CCRIS details table: "
                    "1. 'total_outstanding_balance': The total outstanding balance value from the summary row at the bottom of the table, right before the subheading 'Special Attention Account'. "
                    "2. 'total_limit': The total limit value from the summary row at the bottom of the table, right before the subheading 'Special Attention Account'. "
                    "3. 'ccris_conduct': For each loan row, extract the values (the numeric digits in the monthly columns under the column 'Conduct of Account For Last 12 Months'). There may be multiple loan rows. For each loan row, collect the values into a list of integers. For example, if there are two rows, with the first loan row having all 12 subcolumns populated with the digits shown and the second loan row having only 11 subcolumns populated with the digits shown, then the final ccris_conduct is [[0,0,1,0,0,0,0,0,2,0,0,0], [0,0,1,0,0,0,0,0,2,0,0,0]]. Therefore, if you see a missing month, skip it. Do not represent a missing month with a 0. If you are unsure of the individual digits extracted, then return null for ccris_conduct."
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
                    "3. 'ccris_conduct': For each loan row, extract the values (the numeric digits in the monthly columns under the column 'Conduct of Account For Last 12 Months'). There may be multiple loan rows. For each loan row, collect the values into a list of integers. For example, if there are two rows, with the first loan row having all 12 subcolumns populated with the digits shown and the second loan row having only 11 subcolumns populated with the digits shown, then the final ccris_conduct is [[0,0,1,0,0,0,0,0,2,0,0,0], [0,0,1,0,0,0,0,0,2,0,0,0]]. Therefore, if you see a missing month, skip it. Do not represent a missing month with a 0. If you are unsure of the individual digits extracted, then return null for ccris_conduct."
                    "Return the extracted data in the following JSON format: "
                    "{\"total_outstanding_balance\": value or null, \"total_limit\": value or null, \"ccris_conduct\": [list of conduct strings] or null}."
                )
            case 'credit_info_at_a_glance':
                if self.report_type == ReportType.INDIVIDUAL:
                    return (
                        "Extract the following fields from the table with the heading 'Credit Info at a Glance'. There are three columns: 'Credit Info', 'Source', 'Value'. We are only interested in the first column which shows the field names, and the third column 'Value' which shows the values for the entity. Extract the value for the field 'Bankruptcy Proceedings Record'. Extract the number of legal records in past 24 months (personal capacity) which is the first subrow for the field 'legal records in past 24 months (personal capacity)'. Extract the number of legal records in past 24 months (non-personal capacity) which is the first subrow for the field 'legal records in past 24 months (non-personal capacity)'. Extract the value for the field 'Special Attention Accounts'. If any of these fields are not present in the table, return null for that field. Return the extracted data in the following JSON format: {\"bankruptcy\": value or null, \"legal_personal\": value or null, \"legal_non_personal\": value or null, \"special_attention_accounts\": value or null}."
                    )
                elif self.report_type == ReportType.COMPANY:
                    return (
                        "Extract the following fields from the table with the heading 'Credit Info at a Glance'. There are four columns: 'Credit Info', 'Source', 'Entity', 'Related Parties'. We are only interested in the first column which shows the field names, and the third column 'Entity' which shows the values for the entity. Extract the Entity value for the field 'Winding Up / Bankruptcy Proceedings Record'. Extract the Entity's number of legal records in past 24 months (personal capacity) which is the first subrow for the field 'legal records in past 24 months (personal capacity)'. Extract the Entity's number of legal records in past 24 months (non-personal capacity) which is the first subrow for the field 'legal records in past 24 months (non-personal capacity)'. Extract the Entity value for the field 'Special Attention Accounts'. If any of these fields are not present in the table, return null for that field. Return the extracted data in the following JSON format: {\"bankruptcy_entity\": value or null, \"legal_personal_entity\": value or null, \"legal_non_personal_entity\": value or null, \"special_attention_accounts_entity\": value or null}."
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
                    "Ignore asterisks around values if present. If the value is 0, it may be represented as a dash '-' or an en-dash '–' or an em-dash '—'. If the value is 0.00, return 0.00 and do not return null. Only return null if the field is not present in all the tables in all the images attached. "
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
    
    def __calculate_years__(self, date_str: str) -> int:
        """Calculates years since date string DD-MM-YYYY."""
        try:
            date = datetime.strptime(date_str, '%d-%m-%Y')
            today = datetime.today()
            years_elapsed = today.year - date.year - ((today.month, today.day) < (date.month, date.day))
            return years_elapsed
        except ValueError:
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
        
    def __str_to_decimal__(self, value: str) -> Decimal:
        """Converts a string representation of a number to Decimal, handling commas and spaces."""
        try:
            clean_value = value.replace(',', '').replace(' ', '').replace('%', '').replace('*', '')
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
        threes_to_fives = 0
        high_non_zeroes = 0
        digits = len(flat_list)
        for digit in flat_list:
            if digit == 0:
                zeroes += 1
            elif digit == 1:
                ones += 1
            elif digit == 2:
                twos += 1
            elif 3 <= digit <= 5:
                threes_to_fives += 1
            elif digit >= 6:
                high_non_zeroes += 1
            else:
                digits -= 1  # invalid
        non_zeroes = digits - zeroes
        if digits == zeroes or ((non_zeroes / digits) < 0.2 and non_zeroes == ones):
            return 'Satisfactory'
        elif (non_zeroes / digits) < 0.3 and non_zeroes == (ones + twos):
            return 'Moderate'
        else:
            return 'Poor'

    def __parse_conduct_values__(self, conduct_values: List[str]) -> str:
        """Evaluate conduct of account based on conduct values extracted from CCRIS Details table."""
        digits = 0
        zeroes = 0
        non_zeroes = 0
        ones = 0
        twos = 0
        # TODO check ranges for credit scoring form
        # This is assuming guarantor will never have >9 months lapses in payments...so must double check with gpt4o
        threes_to_fives = 0
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
                    elif char in ['3', '4', '5']:
                        threes_to_fives += 1
                    elif char in ['6', '7', '8', '9']:
                        high_non_zeroes += 1
                    else:
                        non_zeroes -= 1
                        digits -= 1  # invalid character, do not count
        if digits == zeroes or ((non_zeroes / digits) < 0.2 and non_zeroes == ones):
            return 'Satisfactory'
        elif (non_zeroes / digits) < 0.3 and non_zeroes == (ones + twos):
            return 'Moderate'
        else:
            return 'Poor'

    def __extract_conduct__(self, table, start_row_idx: int, end_row_idx: int) -> List[str]:
        """Extracts conduct information from CCRIS Details table."""
        conduct_values = []
        for r_idx in range(start_row_idx, end_row_idx):
            for c_idx in range(11, 23):
                cell = table[r_idx].get(c_idx, "")
                if cell:
                    cell = re.sub(r'\s+', '', cell.strip()) # remove all whitespace
                    conduct_values.append(cell)
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