from dash import Dash, html, dcc, callback, Output, Input, State, clientside_callback
import dash_bootstrap_components as dbc
from datetime import datetime
import pandas as pd
import json

# 导入你现有的模块
from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, KL_TYPE
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


# 初始化Dash应用
app = Dash(
    __name__, 
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# 配置选项的下拉菜单
kl_type_options = [
    {"label": "日K线", "value": "K_DAY"},
    {"label": "周K线", "value": "K_WEEK"},
    {"label": "月K线", "value": "K_MON"},
    {"label": "5分钟", "value": "K_5M"},
    {"label": "15分钟", "value": "K_15M"},
    {"label": "30分钟", "value": "K_30M"},
    {"label": "60分钟", "value": "K_60M"},
]

# 获取默认配置值
default_stock_code = "sh.000300"
default_start_date = datetime(2024, 1, 1).strftime("%Y-%m-%d")
default_end_date = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
default_kl_type = "K_DAY"
default_plot_options = [
    "plot_kline",
    "plot_kline_combine",
    "plot_bi",
    "plot_seg",
    "plot_zs",
    "plot_macd",
    "plot_bsp",
    "plot_rsi",
    "plot_kdj",
]

# 应用布局
app.layout = html.Div(
    [
        # 存储用户输入状态 - 使用local storage代替session
        dcc.Store(id='local-storage', storage_type='local'),
        
        # 添加一个隐藏的div，用于初始化回调
        html.Div(id='init-load', style={'display': 'none'}),
        
        # 主容器 - 设置最大宽度并居中
        dbc.Container(
            [
                html.H1("缠论分析", className="text-center my-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("股票代码设置"),
                                dbc.Input(
                                    id="stock-code",
                                    placeholder="输入股票代码，例如: sh.000300",
                                    value=default_stock_code,
                                    type="text",
                                    persistence=True,
                                    persistence_type='local',
                                ),
                                html.Small(
                                    "支持市场代码，如sh.000001（上证指数）、MSFT等"
                                ),
                            ],
                            width=12,
                            lg=6,
                        ),
                        dbc.Col(
                            [
                                html.H4("日期范围"),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Label("开始日期"),
                                                dcc.DatePickerSingle(
                                                    id="start-date",
                                                    date=default_start_date,
                                                    display_format="YYYY-MM-DD",
                                                    persistence=True,
                                                    persistence_type='local',
                                                ),
                                            ],
                                            width=6,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Label("结束日期"),
                                                dcc.DatePickerSingle(
                                                    id="end-date",
                                                    date=default_end_date,
                                                    display_format="YYYY-MM-DD",
                                                    persistence=True,
                                                    persistence_type='local',
                                                ),
                                            ],
                                            width=6,
                                        ),
                                    ]
                                ),
                            ],
                            width=12,
                            lg=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.H4("K线类型"),
                                dcc.Dropdown(
                                    id="kl-type",
                                    options=kl_type_options,
                                    value=default_kl_type,
                                    clearable=False,
                                    persistence=True,
                                    persistence_type='local',
                                ),
                            ],
                            width=12,
                            lg=6,
                        ),
                        dbc.Col(
                            [
                                html.H4("分析选项"),
                                dbc.Checklist(
                                    id="plot-options",
                                    options=[
                                        {"label": "K线", "value": "plot_kline"},
                                        {"label": "合并K线", "value": "plot_kline_combine"},
                                        {"label": "笔", "value": "plot_bi"},
                                        {"label": "线段", "value": "plot_seg"},
                                        {"label": "中枢", "value": "plot_zs"},
                                        {"label": "MACD", "value": "plot_macd"},
                                        {"label": "买卖点", "value": "plot_bsp"},
                                        {"label": "RSI", "value": "plot_rsi"},
                                        {"label": "KDJ", "value": "plot_kdj"},
                                    ],
                                    value=default_plot_options,
                                    inline=True,
                                    persistence=True,
                                    persistence_type='local',
                                ),
                            ],
                            width=12,
                            lg=6,
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Button(
                                    "生成图表",
                                    id="submit-button",
                                    color="primary",
                                    size="lg",
                                    className="w-100",
                                ),
                            ],
                            width={"size": 6, "offset": 3},  # 按钮居中
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [html.Div(id="plot-container")], 
                            width=12
                        ),
                    ]
                ),
            ],
            className="px-4",
            style={"maxWidth": "1200px", "margin": "0 auto"},  # 设置最大宽度并居中
        ),
    ],
    style={"display": "flex", "justifyContent": "center"}  # 确保整体居中
)


# 使用内置的持久化功能，而不是循环依赖的回调方法
# 每个输入组件都添加了persistence=True和persistence_type='local'属性


@app.callback(
    Output("plot-container", "children"),
    Input("submit-button", "n_clicks"),
    [
        State("stock-code", "value"),
        State("start-date", "date"),
        State("kl-type", "value"),
        State("plot-options", "value"),
        State("end-date", "date"),
    ],
    prevent_initial_call=True,
)
def update_plot(n_clicks, code, begin_time, kl_type_str, plot_options, end_time=None):
    if n_clicks is None:
        return html.Div("请点击生成图表按钮来查看结果", style={"textAlign": "center", "marginTop": "20px"})

    # 将字符串转换为枚举类型
    kl_type = getattr(KL_TYPE, kl_type_str)
    lv_list = [kl_type]

    # 设置绘图配置
    plot_config = {
        "plot_kline": False,
        "plot_kline_combine": False,
        "plot_bi": False,
        "plot_seg": False,
        "plot_eigen": False,
        "plot_zs": False,
        "plot_macd": False,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": False,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
    }

    # 更新用户选择的绘图选项
    for option in plot_options:
        plot_config[option] = True

    # 设置缠论配置
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
            "cal_rsi": "plot_rsi" in plot_options,
            "cal_kdj": "plot_kdj" in plot_options,
            "bs_type": "1,2,3a,1p,2s,3b",
            "print_warning": True,
            "zs_algo": "normal",
        }
    )

    # 绘图参数设置
    plot_para = {
        "seg": {},
        "bi": {},
    }

    try:
        # 实例化CChan分析对象
        chan = CChan(
            code=code,
            begin_time=begin_time,
            end_time=end_time,
            lv_list=lv_list,
            config=config,
            autype=AUTYPE.QFQ,
        )

        # 获取股票名称
        if len(code) >= 5:
            code_name = chan.get_stock_name()
        else:
            code_name = code

        # 创建绘图驱动
        plot_driver = CPlotlyDriver(
            chan,
            plot_config=plot_config,
            plot_para=plot_para,
        )

        # 生成文件名
        if end_time is None:
            end_str = "now"
        else:
            end_str = end_time

        filename = f"{code_name}_{convert_type(lv_list[0])}_{begin_time}_to_{end_str}"

        # 获取图表
        fig = plot_driver.figure

        # 返回图表组件
        return dcc.Graph(id="stock-graph", figure=fig, style={"height": "800px"})

    except Exception as e:
        return html.Div(
            [html.H3("发生错误"), html.P(f"错误信息: {str(e)}")],
            className="text-danger text-center",
        )


if __name__ == "__main__":
    app.run_server(debug=True)