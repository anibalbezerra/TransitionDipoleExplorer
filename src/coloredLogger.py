import logging
import colorlog

# Define a new TRACE log level (between DEBUG and INFO)
TRACE_LEVEL_NUM = 1  # Lower than DEBUG
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def trace(self, message, *args, **kwargs):
    """Custom trace method for loggers"""
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

# Add the trace method to all loggers
logging.Logger.trace = trace

class color_log:
    
    def __init__(self, level) -> None:
        self.logger = self.setup_logger(level)
        

    def setup_logger(self, level):
        # Create a log formatter with color
        log_format = ('%(log_color)s%(levelname)s: %(message)s')        
        color_formatter = colorlog.ColoredFormatter(# Define the color mapping for each log level
            log_format, log_colors={'DEBUG': 'cyan', 'INFO': 'green','WARNING': 'yellow','ERROR': 'red', 'CRITICAL': 'bold_red','TRACE': 'blue', } )    
        logger = logging.getLogger() # Create a logger instance    

        # Create a logger instance
        logger = logging.getLogger()

        if level == 'info':
            logger.setLevel(logging.INFO) # Set the minimum log level (DEBUG, INFO, WARNING, etc.) 
        elif level == 'debug':  
            logger.setLevel(logging.DEBUG) # Set the minimum log level (DEBUG, INFO, WARNING, etc.)
        elif level == 'trace':
            logger.setLevel(TRACE_LEVEL_NUM)  # Custom TRACE level
        
        console_handler = logging.StreamHandler() # Create a console handler and set its log level
        console_handler.setLevel(logging.DEBUG)        
        console_handler.setFormatter(color_formatter)# Set the formatter to the console handler    
        # Add the console handler to the logger
        if not logger.hasHandlers():
            logger.addHandler(console_handler)
        return logger

    # Define a custom trace method
    def trace(self, message, *args, **kwargs):
        if self.logger.isEnabledFor(TRACE_LEVEL_NUM):
            self.logger.log(TRACE_LEVEL_NUM, message, *args, **kwargs)

