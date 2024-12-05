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
    """
    A class to handle colored logging with a custom TRACE level.
    """
    
    def __init__(self, level) -> None:
        """
        Initialize the color_log class with a specified log level.

        :param level: The log level to set ('info', 'debug', 'trace').
        :type level: str
        """
        self.logger = self.setup_logger(level)
        
    def setup_logger(self, level):
        """
        Set up the logger with the specified log level and colored formatting.

        :param level: The log level to set ('info', 'debug', 'trace').
        :type level: str
        :return: The configured logger instance.
        :rtype: logging.Logger
        """
        # Create a log formatter with color
        log_format = ('%(log_color)s%(levelname)s: %(message)s')        
        color_formatter = colorlog.ColoredFormatter(
            log_format, 
            log_colors={
                'DEBUG': 'cyan', 
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red', 
                'CRITICAL': 'bold_red',
                'TRACE': 'blue', 
            }
        )    
        logger = logging.getLogger()  # Create a logger instance    

        # Set the minimum log level based on the input
        if level == 'info':
            logger.setLevel(logging.INFO)  # Set the minimum log level to INFO
        elif level == 'debug':  
            logger.setLevel(logging.DEBUG)  # Set the minimum log level to DEBUG
        elif level == 'trace':
            logger.setLevel(TRACE_LEVEL_NUM)  # Set the minimum log level to custom TRACE level
        
        console_handler = logging.StreamHandler()  # Create a console handler and set its log level
        console_handler.setLevel(logging.DEBUG)        
        console_handler.setFormatter(color_formatter)  # Set the formatter to the console handler    
        
        # Add the console handler to the logger if it doesn't already have handlers
        if not logger.hasHandlers():
            logger.addHandler(console_handler)
        
        return logger

    def trace(self, message, *args, **kwargs):
        """
        Log a message with the TRACE level.

        :param message: The message to log.
        :type message: str
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        """
        if self.logger.isEnabledFor(TRACE_LEVEL_NUM):
            self.logger.log(TRACE_LEVEL_NUM, message, *args, **kwargs)
