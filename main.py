from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver
from Plot.PlotlyDriver import CPlotlyDriver


def convert_type(k_type: KL_TYPE) -> str:
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
    return _dict[k_type]


if __name__ == "__main__":
    # code = "MSTR"
    # 综合指数，例如：sh.000001 上证指数，sz.399106 深证综指 等；
    # 规模指数，例如：sh.000016 上证50，sh.000300 沪深300，sh.000905 中证500，sz.399001 深证成指等；
    code = "sh.000905"
    begin_time = "2023-01-01"
    end_time = None
    lv_list = [KL_TYPE.K_DAY]

    config = CChanConfig(
        {
            "bi_strict": True,
            "trigger_step": False,
            "skip_step": 0,
            "divergence_rate": float("inf"),
            "bsp2_follow_1": True,
            "bsp3_follow_1": True,
            "min_zs_cnt": 0,
            "bs1_peak": True,
            "macd_algo": "peak",
            "cal_rsi": True,
            "cal_kdj": True,
            "bs_type": "1,2,3a,1p,2s,3b",
            "print_warning": True,
            "zs_algo": "normal",
        }
    )

    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": False,
        "plot_zs": True,
        "plot_macd": True,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": True,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": True,
        "plot_kdj": True,
    }

    if config.trigger_step:
        plot_config["plot_rsi"] = False
        plot_config["plot_kdj"] = False

    plot_para = {
        "seg": {
            # "plot_trendline": True,
        },
        "bi": {
            # "show_num": True,
            # "disp_end": True,
        },
        # "figure": {
        #     "x_range": 200,
        # },
        "marker": {
            # "markers": {  # text, position, color
            #     '2023/06/01': ('marker here', 'up', 'red'),
            #     '2023/06/08': ('marker here', 'down')
            # },
        },
    }
    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        # data_src=DATA_SRC.YFINANCE,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,
    )

    if len(code) >= 5:
        code_name = chan.get_stock_name()
    else:
        code_name = code

    if end_time is None:
        end_str = "now"
    else:
        end_str = end_time

    filename = (
        code_name + "_" + convert_type(lv_list[0]) + "_" + begin_time + "_to_" + end_str
    )
    if not config.trigger_step:
        plot_driver = CPlotlyDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )

        plot_driver.figure.write_html("plot_" + filename + ".html")
        print("saved figure.html")
    else:
        CAnimateDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
            output_filename="video_" + filename + ".mp4",
        )
