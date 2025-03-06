import yfinance as yf
import pandas as pd
from datetime import datetime

from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.CTime import CTime
from Common.func_util import kltype_lt_day, str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi


def time_to_ctime(timestamp):
    """Convert pandas Timestamp to CTime object"""
    dt = pd.to_datetime(timestamp)
    return CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute)


def create_item_dict(row, column_mapping):
    """Create dict from yfinance data row with appropriate field mappings"""
    result = {}
    for yf_field, our_field in column_mapping.items():
        if yf_field == 'Date':
            result[our_field] = time_to_ctime(row.name)
        else:
            result[our_field] = float(row[yf_field]) if pd.notna(row[yf_field]) else 0.0
    return result


class CYFinance(CCommonStockApi):
    """Implementation of CommonStockApi for yfinance"""
    
    is_connect = True  # yfinance doesn't require login/connection management
    
    def __init__(
        self,
        code,
        k_type=KL_TYPE.K_DAY,
        begin_date=None,
        end_date=None,
        autype=AUTYPE.QFQ,
    ):
        # For US stocks, we need to ensure no exchange prefix is in the code
        if '.' in code:
            # Convert something like "AAPL.US" to just "AAPL"
            self.orig_code = code
            code = code.split('.')[0]
        
        super(CYFinance, self).__init__(code, k_type, begin_date, end_date, autype)
    
    def get_kl_data(self):
        # Define field mappings between yfinance and our internal fields
        # Intraday data
        column_mapping = {
            'Date': DATA_FIELD.FIELD_TIME,
            'Open': DATA_FIELD.FIELD_OPEN,
            'High': DATA_FIELD.FIELD_HIGH,
            'Low': DATA_FIELD.FIELD_LOW,
            'Close': DATA_FIELD.FIELD_CLOSE,
            'Volume': DATA_FIELD.FIELD_VOLUME,
        }
        
        # Get data from yfinance
        try:
            ticker = yf.Ticker(self.code)
            
            # Convert date format if provided
            start = self.begin_date if not self.begin_date else datetime.strptime(self.begin_date, '%Y-%m-%d')
            end = self.end_date if not self.end_date else datetime.strptime(self.end_date, '%Y-%m-%d')
            
            # Handle adjustment type
            auto_adjust = True if self.autype == AUTYPE.QFQ else False
            back_adjust = True if self.autype == AUTYPE.HFQ else False
            
            # Get historical data
            data = ticker.history(
                period="max" if start is None else None,
                start=start,
                end=end,
                interval=self.__convert_type(),
                auto_adjust=auto_adjust,
                back_adjust=back_adjust
            )
            
            # Process each row of data
            for idx, row in data.iterrows():
                yield CKLine_Unit(create_item_dict(row, column_mapping))
                
        except Exception as e:
            raise Exception(f"Failed to get data from yfinance: {str(e)}")
    
    def SetBasicInfo(self):
        try:
            ticker = yf.Ticker(self.code)
            info = ticker.info
            
            # Set name and type
            self.name = info.get('shortName', self.code)
            
            # Determine if it's a stock or index
            # yfinance doesn't have a direct "type" field, so use quoteType
            quote_type = info.get('quoteType', '').lower()
            self.is_stock = quote_type in ['equity', 'stock'] and quote_type != 'index'
            
        except Exception as e:
            raise Exception(f"Failed to get basic info from yfinance: {str(e)}")
    
    @classmethod
    def do_init(cls):
        # yfinance doesn't need initialization
        pass
    
    @classmethod
    def do_close(cls):
        # yfinance doesn't need closing
        pass
    
    def __convert_type(self):
        """Convert internal k_type to yfinance interval"""
        _dict = {
            KL_TYPE.K_DAY: "1d",
            KL_TYPE.K_WEEK: "1wk",
            KL_TYPE.K_MON: "1mo",
            KL_TYPE.K_5M: "5m",
            KL_TYPE.K_15M: "15m",
            KL_TYPE.K_30M: "30m",
            KL_TYPE.K_60M: "60m",
        }
        return _dict[self.k_type]