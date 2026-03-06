import os
import logging

class Configuration:
    def __init__(self):
        # We no longer need to call load() or use credentials 
        # because the OS environment already has the data.
        pass

    def get_value(self, key: str, default: str = None) -> str:
        if key is None:
            raise Exception('The key parameter is required for get_value().')
        
        # Directly pull from the Environment Variables you just added
        value = os.environ.get(key)
        
        if value is not None:
            return value
        return default

    # Keep your helper functions for lists and booleans
    def read_env_variable(self, var_name, default=None):
        value = self.get_value(var_name, default)
        return value.strip() if value else default

    def read_env_boolean(self, var_name, default=False):
        value = self.get_value(var_name, str(default)).strip().lower()
        return value in ['true', '1', 'yes']