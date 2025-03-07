import inspect
from typing import Dict, List, Literal, Optional, Tuple, Union
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Chan import CChan
from Common.CEnum import BI_DIR, FX_TYPE, KL_TYPE, KLINE_DIR, TREND_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Math.Demark import T_DEMARK_INDEX, CDemarkEngine

from .PlotMeta import CBi_meta, CChanPlotMeta, CZS_meta


def reformat_plot_config(plot_config: Dict[str, bool]):
    """
    兼容不填写`plot_`前缀的情况
    """

    def _format(s):
        return s if s.startswith("plot_") else f"plot_{s}"

    return {_format(k): v for k, v in plot_config.items()}


def parse_single_lv_plot_config(plot_config: Union[str, dict, list]) -> Dict[str, bool]:
    """
    返回单一级别的plot_config配置
    """
    if isinstance(plot_config, dict):
        return reformat_plot_config(plot_config)
    elif isinstance(plot_config, str):
        return reformat_plot_config(
            dict([(k.strip().lower(), True) for k in plot_config.split(",")])
        )
    elif isinstance(plot_config, list):
        return reformat_plot_config(
            dict([(k.strip().lower(), True) for k in plot_config])
        )
    else:
        raise CChanException("plot_config only support list/str/dict", ErrCode.PLOT_ERR)


def parse_plot_config(
    plot_config: Union[str, dict, list], lv_list: List[KL_TYPE]
) -> Dict[KL_TYPE, Dict[str, bool]]:
    """
    支持：
        - 传入字典
        - 传入字符串，逗号分割
        - 传入数组，元素为各个需要画的笔的元素
        - 传入key为各个级别的字典
        - 传入key为各个级别的字符串
        - 传入key为各个级别的数组
    """
    if isinstance(plot_config, dict):
        if all(isinstance(_key, str) for _key in plot_config.keys()):  # 单层字典
            return {lv: parse_single_lv_plot_config(plot_config) for lv in lv_list}
        elif all(
            isinstance(_key, KL_TYPE) for _key in plot_config.keys()
        ):  # key为KL_TYPE
            for lv in lv_list:
                assert lv in plot_config
            return {lv: parse_single_lv_plot_config(plot_config[lv]) for lv in lv_list}
        else:
            raise CChanException(
                "plot_config if is dict, key must be str/KL_TYPE", ErrCode.PLOT_ERR
            )
    return {lv: parse_single_lv_plot_config(plot_config) for lv in lv_list}


def set_x_tick(fig, row, col, x_limits, tick, x_tick_num: int):
    """
    设置x轴刻度，plotly版本
    """
    assert x_tick_num > 1
    # 计算刻度位置
    step = max(1, int((x_limits[1] - x_limits[0]) / float(x_tick_num)))
    tick_positions = list(range(x_limits[0], x_limits[1] + 1, step))
    tick_texts = [tick[i] for i in tick_positions]

    # 设置x轴范围和刻度
    fig.update_xaxes(
        range=[x_limits[0], x_limits[1] + 1],
        tickvals=tick_positions,
        ticktext=tick_texts,
        tickangle=20,
        row=row,
        col=col,
    )


def cal_y_range(meta: CChanPlotMeta, x_begin):
    """
    计算Y轴范围，plotly版本
    """
    y_min = float("inf")
    y_max = float("-inf")
    for klc_meta in meta.klc_list:
        if klc_meta.klu_list[-1].idx < x_begin:
            continue  # 不绘制范围外的
        if klc_meta.high > y_max:
            y_max = klc_meta.high
        if klc_meta.low < y_min:
            y_min = klc_meta.low
    return (y_min, y_max)


def create_figure(
    figure_config,
    lv_lst: List[KL_TYPE],
    plot_count: int,
) -> Tuple[go.Figure, Dict[KL_TYPE, List[Tuple[int, int]]]]:
    """
    返回：
        - Figure
        - Dict[KL_TYPE, List[Tuple[int, int]]]: 每个级别对应的(row, col)位置列表，如果长度为1，说明不需要画macd，否则需要
    """
    macd_h_ratio = figure_config.get("macd_h", 0.25)

    # 计算总行数和每个子图的行高比例
    rows = 1
    row_heights = [1]
    if plot_count >= 1:
        rows += plot_count
        row_heights.extend([macd_h_ratio] * plot_count)

    # 创建子图
    fig = make_subplots(
        rows=rows,
        cols=1,
        row_heights=row_heights,
        vertical_spacing=0.03,
        shared_xaxes=True,
    )

    # 设置图表大小
    default_w, default_h = 24, 50
    w = figure_config.get("w", default_w)
    h = figure_config.get("h", default_h)
    total_h = h * sum(row_heights) / len(row_heights)  # 按比例计算总高度

    if rows == 1:
        fig.update_layout(width=w * 50, height=total_h * 50)
    elif rows == 2:
        fig.update_layout(
            width=w * 50,
            height=total_h * 50,
            xaxis=dict(rangeslider=dict(visible=False)),
            xaxis2=dict(
                matches="x", rangeslider=dict(visible=True, thickness=0.05)
            ),  # This links the second x-axis to the first one
        )  # plotly单位与matplotlib不同，需要转换
        # 对第一个子图启用尖刺线
    elif rows == 3:
        fig.update_layout(
            width=w * 50,
            height=total_h * 50,
            xaxis=dict(rangeslider=dict(visible=False)),
            xaxis2=dict(matches="x", rangeslider=dict(visible=False)),
            xaxis3=dict(matches="x", rangeslider=dict(visible=True, thickness=0.05)),
        )

    elif rows == 4:
        fig.update_layout(
            width=w * 50,
            height=total_h * 50,
            xaxis=dict(rangeslider=dict(visible=False)),
            xaxis2=dict(rangeslider=dict(visible=False)),
            xaxis3=dict(rangeslider=dict(visible=False)),
            xaxis4=dict(rangeslider=dict(visible=True, thickness=0.05)),
        )

        # fig.update_yaxes(range=[0, 100], row=2, col=1)

    # 关键设置：hovermode和spike线设置
    fig.update_layout(
        hovermode="x unified",  # 显示统一的垂直线
        # hoverdistance=100,  # 增加hover的捕获距离
        # spikedistance=1000,  # 增加spike线的显示距离
    )

    # 为所有子图设置相同的spike线属性
    for i in range(1, rows):  # 遍历三个子图
        fig.update_xaxes(
            showspikes=True,
            spikemode="across",  # 让spike线穿过整个图表
            spikesnap="cursor",  # 让spike线跟随光标
            spikecolor="gray",
            spikethickness=1,
            spikedash="dot",
            row=i,
            col=1,
        )

        fig.update_yaxes(
            showspikes=True,  # 同样为Y轴添加spike线
            spikemode="across",
            spikesnap="cursor",
            spikecolor="gray",
            spikethickness=1,
            spikedash="dot",
            row=i,
            col=1,
        )

    # 设置hover模式
    fig.update_layout(hovermode="x unified")

    # 为每个级别分配对应的行
    axes_dict = {}
    current_row = 1
    for lv in lv_lst:
        if plot_count == 1:
            axes_dict[lv] = [(current_row, 1), (current_row + 1, 1)]
            current_row += 2
        elif plot_count == 2:
            axes_dict[lv] = [
                (current_row, 1),
                (current_row + 1, 1),
                (current_row + 2, 1),
            ]
            current_row += 3
        elif plot_count == 3:
            axes_dict[lv] = [
                (current_row, 1),
                (current_row + 1, 1),
                (current_row + 2, 1),
                (current_row + 3, 1),
            ]
            current_row += 4
        else:
            # 只有主图的行号
            axes_dict[lv] = [(current_row, 1)]
            current_row += 1

    return fig, axes_dict


def cal_x_limit(meta: CChanPlotMeta, x_range):
    """
    计算X轴范围
    """
    X_LEN = meta.klu_len
    return (
        [X_LEN - x_range, X_LEN - 1] if x_range and X_LEN > x_range else [0, X_LEN - 1]
    )


def set_grid(fig, row, col, config):
    """
    设置网格，plotly版本
    """
    if config is None:
        return

    grid_params = dict(row=row, col=col)

    if config == "xy":
        grid_params.update(showgrid=True)
        fig.update_xaxes(**grid_params)
        fig.update_yaxes(**grid_params)
        return

    if config == "x":
        fig.update_xaxes(showgrid=True, **grid_params)
        fig.update_yaxes(showgrid=False, **grid_params)
        return

    if config == "y":
        fig.update_xaxes(showgrid=False, **grid_params)
        fig.update_yaxes(showgrid=True, **grid_params)
        return

    raise CChanException(f"unsupport grid config={config}", ErrCode.PLOT_ERR)


def GetPlotMeta(chan: CChan, figure_config) -> List[CChanPlotMeta]:
    """
    获取绘图元数据
    """
    plot_metas = [CChanPlotMeta(chan[kl_type]) for kl_type in chan.lv_list]
    if figure_config.get("only_top_lv", False):
        plot_metas = [plot_metas[0]]
    return plot_metas


class CPlotlyDriver:
    def __init__(
        self, chan: CChan, plot_config: Union[str, dict, list] = "", plot_para=None
    ):
        if plot_para is None:
            plot_para = {}
        figure_config: dict = plot_para.get("figure", {})

        plot_config = parse_plot_config(plot_config, chan.lv_list)
        plot_metas = GetPlotMeta(chan, figure_config)
        self.lv_lst = chan.lv_list[: len(plot_metas)]

        x_range = self.GetRealXrange(figure_config, plot_metas[0])
        plot_kdj: bool = {
            conf.get("plot_kdj", False) for kl_type, conf in plot_config.items()
        }
        plot_rsi: bool = {
            conf.get("plot_rsi", False) for kl_type, conf in plot_config.items()
        }
        plot_macd: bool = {
            conf.get("plot_macd", False) for kl_type, conf in plot_config.items()
        }

        plot_list = [plot_kdj, plot_rsi, plot_macd]
        plot_count = 0
        for plot in plot_list:
            if plot:
                plot_count += 1

        self.figure, axes_positions = create_figure(
            figure_config, self.lv_lst, plot_count
        )

        sseg_begin = 0
        slv_seg_cnt = plot_para.get("seg", {}).get("sub_lv_cnt", None)
        sbi_begin = 0
        slv_bi_cnt = plot_para.get("bi", {}).get("sub_lv_cnt", None)
        srange_begin = 0
        assert slv_seg_cnt is None or slv_bi_cnt is None, (
            "you can set at most one of seg_sub_lv_cnt/bi_sub_lv_cnt"
        )

        for meta, lv in zip(plot_metas, self.lv_lst):  # type: ignore
            main_pos = axes_positions[lv][0]  # (row, col) for main chart
            # macd_pos = (
            #     None if len(axes_positions[lv]) == 1 else axes_positions[lv][1]
            # )  # (row, col) for MACD

            # 先确认是否需要画RSI
            kdj_pos = None
            rsi_pos = None
            macd_pos = None
            kdj_pos = axes_positions[lv][1]
            rsi_pos = axes_positions[lv][2]
            macd_pos = axes_positions[lv][3]

            # Set grid and title
            set_grid(
                self.figure, main_pos[0], main_pos[1], figure_config.get("grid", "xy")
            )
            self.figure.update_layout(
                title_text=f"{chan.get_stock_name()}/{lv.name.split('K_')[1]}",
                title_font=dict(size=16, color="red"),
                title_x=0,
            )

            x_limits = cal_x_limit(meta, x_range)
            if lv != self.lv_lst[0]:
                if sseg_begin != 0 or sbi_begin != 0:
                    x_limits[0] = max(sseg_begin, sbi_begin)
                elif srange_begin != 0:
                    x_limits[0] = srange_begin

            # Set x ticks for main chart and MACD
            set_x_tick(
                self.figure,
                main_pos[0],
                main_pos[1],
                x_limits,
                meta.datetick,
                figure_config.get("x_tick_num", 10),
            )
            if macd_pos:
                set_x_tick(
                    self.figure,
                    macd_pos[0],
                    macd_pos[1],
                    x_limits,
                    meta.datetick,
                    figure_config.get("x_tick_num", 10),
                )

            # Calculate y range
            self.y_min, self.y_max = cal_y_range(meta, x_limits[0])

            # Draw elements
            self.DrawElement(
                plot_config[lv],
                meta,
                main_pos,
                lv,
                plot_para,
                kdj_pos,
                rsi_pos,
                macd_pos,
                x_limits,
            )

            if lv != self.lv_lst[-1]:
                if slv_seg_cnt is not None:
                    sseg_begin = meta.sub_last_kseg_start_idx(slv_seg_cnt)
                if slv_bi_cnt is not None:
                    sbi_begin = meta.sub_last_kbi_start_idx(slv_bi_cnt)
                if x_range != 0:
                    srange_begin = meta.sub_range_start_idx(x_range)

            # Set y limit for main chart
            self.figure.update_yaxes(
                range=[self.y_min, self.y_max], row=main_pos[0], col=main_pos[1]
            )

    def GetRealXrange(self, figure_config, meta: CChanPlotMeta):
        x_range = figure_config.get("x_range", 0)
        bi_cnt = figure_config.get("x_bi_cnt", 0)
        seg_cnt = figure_config.get("x_seg_cnt", 0)
        x_begin_date = figure_config.get("x_begin_date", 0)
        if x_range != 0:
            assert bi_cnt == 0 and seg_cnt == 0 and x_begin_date == 0, (
                "x_range/x_bi_cnt/x_seg_cnt/x_begin_date can not be set at the same time"
            )
            return x_range
        if bi_cnt != 0:
            assert x_range == 0 and seg_cnt == 0 and x_begin_date == 0, (
                "x_range/x_bi_cnt/x_seg_cnt/x_begin_date can not be set at the same time"
            )
            X_LEN = meta.klu_len
            if len(meta.bi_list) < bi_cnt:
                return 0
            x_range = X_LEN - meta.bi_list[-bi_cnt].begin_x
            return x_range
        if seg_cnt != 0:
            assert x_range == 0 and bi_cnt == 0 and x_begin_date == 0, (
                "x_range/x_bi_cnt/x_seg_cnt/x_begin_date can not be set at the same time"
            )
            X_LEN = meta.klu_len
            if len(meta.seg_list) < seg_cnt:
                return 0
            x_range = X_LEN - meta.seg_list[-seg_cnt].begin_x
            return x_range
        if x_begin_date != 0:
            assert x_range == 0 and bi_cnt == 0 and seg_cnt == 0, (
                "x_range/x_bi_cnt/x_seg_cnt/x_begin_date can not be set at the same time"
            )
            x_range = 0
            for date_tick in meta.datetick[::-1]:
                if date_tick >= x_begin_date:
                    x_range += 1
                else:
                    break
            return x_range
        return x_range

    def DrawElement(
        self,
        plot_config: Dict[str, bool],
        meta: CChanPlotMeta,
        main_pos: Tuple[int, int],
        lv,
        plot_para,
        kdj_pos: Optional[Tuple[int, int]],
        rsi_pos: Optional[Tuple[int, int]],
        macd_pos: Optional[Tuple[int, int]],
        x_limits,
    ):
        if plot_config.get("plot_kline", False):
            self.draw_klu(meta, main_pos, x_limits, **plot_para.get("kl", {}))
        if plot_config.get("plot_kline_combine", False):
            self.draw_klc(meta, main_pos, x_limits, **plot_para.get("klc", {}))
        if plot_config.get("plot_bi", False):
            self.draw_bi(meta, main_pos, x_limits, lv, **plot_para.get("bi", {}))
        if plot_config.get("plot_seg", False):
            self.draw_seg(meta, main_pos, x_limits, lv, **plot_para.get("seg", {}))
        if plot_config.get("plot_segseg", False):
            self.draw_segseg(meta, main_pos, x_limits, **plot_para.get("segseg", {}))
        if plot_config.get("plot_eigen", False):
            self.draw_eigen(meta, main_pos, x_limits, **plot_para.get("eigen", {}))
        if plot_config.get("plot_segeigen", False):
            self.draw_segeigen(
                meta, main_pos, x_limits, **plot_para.get("segeigen", {})
            )
        if plot_config.get("plot_zs", False):
            self.draw_zs(meta, main_pos, x_limits, **plot_para.get("zs", {}))
        if plot_config.get("plot_segzs", False):
            self.draw_segzs(meta, main_pos, x_limits, **plot_para.get("segzs", {}))
        if plot_config.get("plot_macd", False):
            assert macd_pos is not None
            self.draw_macd(meta, macd_pos, x_limits, **plot_para.get("macd", {}))
        if plot_config.get("plot_mean", False):
            self.draw_mean(meta, main_pos, **plot_para.get("mean", {}))
        if plot_config.get("plot_channel", False):
            self.draw_channel(meta, main_pos, **plot_para.get("channel", {}))
        if plot_config.get("plot_boll", False):
            self.draw_boll(meta, main_pos, x_limits, **plot_para.get("boll", {}))
        if plot_config.get("plot_bsp", False):
            self.draw_bs_point(meta, main_pos, x_limits, **plot_para.get("bsp", {}))
        if plot_config.get("plot_segbsp", False):
            self.draw_seg_bs_point(
                meta, main_pos, x_limits, **plot_para.get("seg_bsp", {})
            )
        if plot_config.get("plot_demark", False):
            self.draw_demark(meta, main_pos, x_limits, **plot_para.get("demark", {}))
        if plot_config.get("plot_marker", False):
            self.draw_marker(
                meta, main_pos, x_limits, **plot_para.get("marker", {"markers": {}})
            )
        if plot_config.get("plot_rsi", False):
            self.draw_rsi(meta, rsi_pos, x_limits, **plot_para.get("rsi", {}))
        if plot_config.get("plot_kdj", False):
            self.draw_kdj(meta, kdj_pos, x_limits, **plot_para.get("kdj", {}))

    def ShowDrawFuncHelper(self):
        # 写README的时候显示所有画图函数的参数和默认值
        for func in dir(self):
            if not func.startswith("draw_"):
                continue
            show_func_helper(eval(f"self.{func}"))

    def save2img(self, path):
        self.figure.write_image(path)

    def draw_klu(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        width=0.4,
        rugd=True,
        plot_mode="kl",
    ):
        # rugd: red up green down
        row, col = pos
        up_color = "red" if rugd else "green"
        down_color = "green" if rugd else "red"

        x_begin, x_end = x_limits[0], x_limits[1]

        if plot_mode == "kl":
            # Prepare data for candlestick chart
            x_vals = []
            open_vals = []
            high_vals = []
            low_vals = []
            close_vals = []
            date_vals = []
            customdata = []  # 用于存放涨跌幅
            self.text_vals_klu = []
            self.text_vals_rsi = []
            self.text_vals_kdj = []
            self.text_vals_macd = []

            last_close = 0
            for kl in meta.klu_iter():
                i = kl.idx
                if i + width < x_begin:
                    continue  # Skip points outside plotting range

                if i > x_end:
                    break

                x_vals.append(i)
                date_vals.append(kl.time.to_str())
                open_vals.append(kl.open)
                high_vals.append(kl.high)
                low_vals.append(kl.low)
                close_vals.append(kl.close)
                pct = (kl.close - last_close) / last_close if last_close != 0 else 0
                last_close = kl.close
                # customdata 要求是二维数组，每个元素是一个数组
                customdata.append(pct)
                text_str = (
                    f"日期: {kl.time.to_str()}<br>"
                    f"开: {kl.open:.2f}<br>"
                    f"高: {kl.high:.2f}<br>"
                    f"低: {kl.low:.2f}<br>"
                    f"收: {kl.close:.2f}<br>"
                    f"涨跌幅: {pct:.2%}"
                )
                self.text_vals_klu.append(text_str)

                # MACD, KDJ, RSI
                text_str = (
                    f"日期: {kl.time.to_str()}<br>"
                    f"DIF: {kl.macd.DIF:.2f}<br>"
                    f"DEA: {kl.macd.DEA:.2f}<br>"
                    f"MACD: {kl.macd.macd:.2f}"
                )
                self.text_vals_macd.append(text_str)

                text_str = (
                    f"日期: {kl.time.to_str()}<br>"
                    f"K: {kl.kdj.k:.2f}<br>"
                    f"D: {kl.kdj.d:.2f}<br>"
                    f"J: {kl.kdj.j:.2f}"
                )
                self.text_vals_kdj.append(text_str)

                text_str = (
                    f"日期: {kl.time.to_str()}<br>"
                    f"RSI: {kl.rsi6:.2f}<br>"
                    f"RSI: {kl.rsi12:.2f}<br>"
                    f"RSI: {kl.rsi24:.2f}"
                )

                self.text_vals_rsi.append(text_str)

            if x_vals:  # Only plot if we have data points
                self.figure.add_trace(
                    go.Candlestick(
                        x=x_vals,
                        open=open_vals,
                        high=high_vals,
                        low=low_vals,
                        close=close_vals,
                        increasing_line_color=up_color,
                        decreasing_line_color=down_color,
                        name="K线",
                        # customdata=customdata,
                        hovertext=self.text_vals_klu,  # 传入自定义文本数组
                        hoverinfo="text",  # 指定只显示 text，不显示默认的OHLC
                    ),
                    row=row,
                    col=col,
                )

        else:  # Plot line mode (close, high, low, open)
            x_vals = []
            y_vals = []

            for kl in meta.klu_iter():
                i = kl.idx
                if i + width < x_begin:
                    continue

                x_vals.append(i)
                if plot_mode == "close":
                    y_vals.append(kl.close)
                elif plot_mode == "high":
                    y_vals.append(kl.high)
                elif plot_mode == "low":
                    y_vals.append(kl.low)
                elif plot_mode == "open":
                    y_vals.append(kl.open)
                else:
                    raise CChanException(
                        f"unknow plot mode={plot_mode}, must be one of kl/close/open/high/low",
                        ErrCode.PLOT_ERR,
                    )

            if x_vals:
                self.figure.add_trace(
                    go.Scatter(x=x_vals, y=y_vals, mode="lines", name=plot_mode),
                    row=row,
                    col=col,
                )

    def draw_klc(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        width=0.4,
        plot_single_kl=True,
    ):
        row, col = pos
        color_map = {
            FX_TYPE.TOP: "red",
            FX_TYPE.BOTTOM: "blue",
            KLINE_DIR.UP: "green",
            KLINE_DIR.DOWN: "green",
        }
        x_begin = x_limits[0]

        for klc_meta in meta.klc_list:
            if klc_meta.klu_list[-1].idx + width < x_begin:
                continue  # 不绘制范围外的
            if klc_meta.end_idx == klc_meta.begin_idx and not plot_single_kl:
                continue

            color = color_map[klc_meta.type]

            # Create rectangle shapes for klc
            self.figure.add_shape(
                type="rect",
                x0=klc_meta.begin_idx - width,
                y0=klc_meta.low,
                x1=klc_meta.end_idx + width,
                y1=klc_meta.high,
                line=dict(color=color, width=1),
                fillcolor="rgba(0,0,0,0)",
                row=row,
                col=col,
            )

    def draw_bi(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        lv,
        color="black",
        show_num=False,
        num_fontsize=15,
        num_color="red",
        sub_lv_cnt=None,
        facecolor="green",
        alpha=0.1,
        disp_end=False,
        end_color="black",
        end_fontsize=10,
    ):
        """
        使用 plotly 画出笔的主函数。

        参数说明:
        ----------
        self : 你的类实例（内部含 self.figure, self.y_min, self.y_max 等）
        meta : CChanPlotMeta ，含有 meta.bi_list 等绘图所需数据
        pos : (row, col)，图在 subplot 中的位置
        x_limits : (x_begin, x_end)，x 轴显示范围
        lv : 当前级别
        color : 笔的颜色
        show_num : 是否在笔的中间显示编号
        num_fontsize : 编号字体大小
        num_color : 编号字体颜色
        sub_lv_cnt : 用来高亮显示子级别笔段的数量
        facecolor : 高亮区块的颜色
        alpha : 高亮区块的透明度
        disp_end : 是否显示笔端数值
        end_color : 笔端文字颜色
        end_fontsize : 笔端文字大小
        """
        # 从 pos 解包出 row, col 以便后续在子图里绘制
        row, col = pos

        # x_begin 用于过滤不在视野范围内的 bi
        x_begin = x_limits[0]

        for bi_idx, bi in enumerate(meta.bi_list):
            # 如果笔的结束点都在可视范围左边，就跳过
            if bi.end_x < x_begin:
                continue

            # 1. 使用你新写好的 helper 函数来绘制笔
            plot_bi_element(bi, self.figure, pos, color)

            # 2. 如果需要显示笔的编号，则添加对应注释
            if show_num and bi.begin_x >= x_begin:
                self.figure.add_annotation(
                    x=(bi.begin_x + bi.end_x) / 2,
                    y=(bi.begin_y + bi.end_y) / 2,
                    text=f"{bi.idx}",
                    font=dict(size=num_fontsize, color=num_color),
                    showarrow=False,
                    row=row,
                    col=col,
                )

            # 3. 如果需要显示笔端数值，则使用 helper 函数 bi_text 来添加
            if disp_end:
                bi_text(bi_idx, self.figure, bi, end_fontsize, end_color, pos)

        # 如果需要对子级别笔段进行高亮
        if sub_lv_cnt is not None and len(self.lv_lst) > 1 and lv != self.lv_lst[-1]:
            # 如果子级笔段数量超出当前笔列表长度，则直接返回
            if sub_lv_cnt >= len(meta.bi_list):
                return
            else:
                # 取最后 sub_lv_cnt 条笔的起始 x
                begin_idx = meta.bi_list[-sub_lv_cnt].begin_x

            y_begin, y_end = self.y_min, self.y_max
            x_end = x_limits[1]

            # 将 facecolor 转成 rgb，组装成 plotly 需要的 rgba
            r = int(facecolor[1:3], 16)
            g = int(facecolor[3:5], 16)
            b = int(facecolor[5:7], 16)

            # 绘制高亮覆盖区域
            self.figure.add_trace(
                go.Scatter(
                    x=[begin_idx, begin_idx, x_end, x_end, begin_idx],
                    y=[y_begin, y_end, y_end, y_begin, y_begin],
                    fill="toself",
                    fillcolor=f"rgba({r},{g},{b},{alpha})",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

    def draw_seg(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        lv,
        width=5,
        color="green",
        sub_lv_cnt=None,
        facecolor="green",
        alpha=0.1,
        disp_end=False,
        end_color="green",
        end_fontsize=13,
        plot_trendline=False,
        trendline_color="red",
        trendline_width=3,
        show_num=False,
        num_fontsize=25,
        num_color="blue",
    ):
        row, col = pos
        x_begin = x_limits[0]

        for seg_idx, seg_meta in enumerate(meta.seg_list):
            if seg_meta.end_x < x_begin:
                continue

            # Plot segment line
            line_style = "solid" if seg_meta.is_sure else "dash"
            self.figure.add_trace(
                go.Scatter(
                    x=[seg_meta.begin_x, seg_meta.end_x],
                    y=[seg_meta.begin_y, seg_meta.end_y],
                    mode="lines",
                    line=dict(color=color, width=width, dash=line_style),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            # Display end values if requested
            if disp_end:
                bi_text(seg_idx, self.figure, seg_meta, end_fontsize, end_color, pos)

            # Plot trendlines if requested
            if plot_trendline:
                if seg_meta.tl.get("support"):
                    tl_meta = seg_meta.format_tl(seg_meta.tl["support"])
                    self.figure.add_trace(
                        go.Scatter(
                            x=[tl_meta[0], tl_meta[2]],
                            y=[tl_meta[1], tl_meta[3]],
                            mode="lines",
                            line=dict(color=trendline_color, width=trendline_width),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

                if seg_meta.tl.get("resistance"):
                    tl_meta = seg_meta.format_tl(seg_meta.tl["resistance"])
                    self.figure.add_trace(
                        go.Scatter(
                            x=[tl_meta[0], tl_meta[2]],
                            y=[tl_meta[1], tl_meta[3]],
                            mode="lines",
                            line=dict(color=trendline_color, width=trendline_width),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row,
                        col=col,
                    )

            # Show segment number if requested
            if show_num and seg_meta.begin_x >= x_begin:
                self.figure.add_annotation(
                    x=(seg_meta.begin_x + seg_meta.end_x) / 2,
                    y=(seg_meta.begin_y + seg_meta.end_y) / 2,
                    text=f"{seg_meta.idx}",
                    font=dict(size=num_fontsize, color=num_color),
                    showarrow=False,
                    row=row,
                    col=col,
                )

        # Handle sub-level highlighting
        if sub_lv_cnt is not None and len(self.lv_lst) > 1 and lv != self.lv_lst[-1]:
            if sub_lv_cnt >= len(meta.seg_list):
                return
            else:
                begin_idx = meta.seg_list[-sub_lv_cnt].begin_x

            y_begin, y_end = self.y_min, self.y_max
            x_end = x_limits[1]

            # Create RGB color with alpha for highlighting
            rgb_color = f"rgba({int(facecolor[1:3], 16)},{int(facecolor[3:5], 16)},{int(facecolor[5:7], 16)},{alpha})"

            self.figure.add_trace(
                go.Scatter(
                    x=[begin_idx, begin_idx, x_end, x_end, begin_idx],
                    y=[y_begin, y_end, y_end, y_begin, y_begin],
                    fill="toself",
                    fillcolor=rgb_color,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

    def draw_segseg(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        width=7,
        color="brown",
        disp_end=False,
        end_color="brown",
        end_fontsize=15,
        show_num=False,
        num_fontsize=30,
        num_color="blue",
    ):
        row, col = pos
        x_begin = x_limits[0]

        for seg_idx, seg_meta in enumerate(meta.segseg_list):
            if seg_meta.end_x < x_begin:
                continue

            # Plot segseg line
            line_style = "solid" if seg_meta.is_sure else "dash"
            self.figure.add_trace(
                go.Scatter(
                    x=[seg_meta.begin_x, seg_meta.end_x],
                    y=[seg_meta.begin_y, seg_meta.end_y],
                    mode="lines",
                    line=dict(color=color, width=width, dash=line_style),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

            # Display end values if requested
            if disp_end:
                # Display beginning value for first segment
                if seg_idx == 0:
                    valign = "top" if seg_meta.dir == BI_DIR.UP else "bottom"
                    self.figure.add_annotation(
                        x=seg_meta.begin_x,
                        y=seg_meta.begin_y,
                        text=f"{seg_meta.begin_y:.2f}",
                        font=dict(size=end_fontsize, color=end_color),
                        showarrow=False,
                        yanchor=valign,
                        xanchor="center",
                        row=row,
                        col=col,
                    )

                # Display ending value for all segments
                valign = "top" if seg_meta.dir == BI_DIR.DOWN else "bottom"
                self.figure.add_annotation(
                    x=seg_meta.end_x,
                    y=seg_meta.end_y,
                    text=f"{seg_meta.end_y:.2f}",
                    font=dict(size=end_fontsize, color=end_color),
                    showarrow=False,
                    yanchor=valign,
                    xanchor="center",
                    row=row,
                    col=col,
                )

            # Show segment number if requested
            if show_num and seg_meta.begin_x >= x_begin:
                self.figure.add_annotation(
                    x=(seg_meta.begin_x + seg_meta.end_x) / 2,
                    y=(seg_meta.begin_y + seg_meta.end_y) / 2,
                    text=f"{seg_meta.idx}",
                    font=dict(size=num_fontsize, color=num_color),
                    showarrow=False,
                    row=row,
                    col=col,
                )

    def plot_single_eigen(
        self,
        eigenfx_meta,
        pos: Tuple[int, int],
        x_limits,
        color_top,
        color_bottom,
        alpha,
        only_peak,
    ):
        row, col = pos
        x_begin = x_limits[0]
        color = color_top if eigenfx_meta.fx == FX_TYPE.TOP else color_bottom

        for idx, eigen_meta in enumerate(eigenfx_meta.ele):
            if eigen_meta.begin_x + eigen_meta.w < x_begin:
                continue
            if only_peak and idx != 1:
                continue

            # Add rectangle shape for eigen element
            self.figure.add_shape(
                type="rect",
                x0=eigen_meta.begin_x,
                y0=eigen_meta.begin_y,
                x1=eigen_meta.begin_x + eigen_meta.w,
                y1=eigen_meta.begin_y + eigen_meta.h,
                fillcolor=color,
                opacity=alpha,
                line=dict(width=0),
                row=row,
                col=col,
            )

    def draw_eigen(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        color_top="red",
        color_bottom="blue",
        aplha=0.5,
        only_peak=False,
    ):
        for eigenfx_meta in meta.eigenfx_lst:
            self.plot_single_eigen(
                eigenfx_meta, pos, x_limits, color_top, color_bottom, aplha, only_peak
            )

    def draw_segeigen(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        color_top="red",
        color_bottom="blue",
        aplha=0.5,
        only_peak=False,
    ):
        for eigenfx_meta in meta.seg_eigenfx_lst:
            self.plot_single_eigen(
                eigenfx_meta, pos, x_limits, color_top, color_bottom, aplha, only_peak
            )

    def draw_zs(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        color="orange",
        linewidth=2,
        sub_linewidth=0.5,
        show_text=False,
        fontsize=14,
        text_color="orange",
        draw_one_bi_zs=False,
    ):
        row, col = pos
        linewidth = max(linewidth, 2)
        x_begin = x_limits[0]

        for zs_meta in meta.zs_lst:
            if not draw_one_bi_zs and zs_meta.is_onebi_zs:
                continue
            if zs_meta.begin + zs_meta.w < x_begin:
                continue

            # 创建主ZS矩形
            line_style = "solid" if zs_meta.is_sure else "dash"
            self.figure.add_shape(
                type="rect",
                x0=zs_meta.begin,
                y0=zs_meta.low,
                x1=zs_meta.begin + zs_meta.w,
                y1=zs_meta.low + zs_meta.h,
                line=dict(color=color, width=linewidth, dash=line_style),
                fillcolor="rgba(0,0,0,0)",
                row=row,
                col=col,
            )

            # 创建子ZS矩形
            for sub_zs_meta in zs_meta.sub_zs_lst:
                self.figure.add_shape(
                    type="rect",
                    x0=sub_zs_meta.begin,
                    y0=sub_zs_meta.low,
                    x1=sub_zs_meta.begin + sub_zs_meta.w,
                    y1=sub_zs_meta.low + sub_zs_meta.h,
                    line=dict(color=color, width=sub_linewidth, dash=line_style),
                    fillcolor="rgba(0,0,0,0)",
                    row=row,
                    col=col,
                )

            # 如果需要显示文本，则调用 helper 函数添加文本
            if show_text:
                add_zs_text(self.figure, zs_meta, fontsize, text_color, (row, col))
                for sub_zs_meta in zs_meta.sub_zs_lst:
                    add_zs_text(
                        self.figure, sub_zs_meta, fontsize, text_color, (row, col)
                    )

    def draw_segzs(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        color="red",
        linewidth=10,
        sub_linewidth=4,
    ):
        row, col = pos
        linewidth = max(linewidth, 2)
        x_begin = x_limits[0]

        for zs_meta in meta.segzs_lst:
            if zs_meta.begin + zs_meta.w < x_begin:
                continue

            line_style = "solid" if zs_meta.is_sure else "dash"
            self.figure.add_shape(
                type="rect",
                x0=zs_meta.begin,
                y0=zs_meta.low,
                x1=zs_meta.begin + zs_meta.w,
                y1=zs_meta.low + zs_meta.h,
                line=dict(color=color, width=linewidth, dash=line_style),
                fillcolor="rgba(0,0,0,0)",
                row=row,
                col=col,
            )

            for sub_zs_meta in zs_meta.sub_zs_lst:
                self.figure.add_shape(
                    type="rect",
                    x0=sub_zs_meta.begin,
                    y0=sub_zs_meta.low,
                    x1=sub_zs_meta.begin + sub_zs_meta.w,
                    y1=sub_zs_meta.low + sub_zs_meta.h,
                    line=dict(color=color, width=sub_linewidth, dash=line_style),
                    fillcolor="rgba(0,0,0,0)",
                    row=row,
                    col=col,
                )

    def draw_macd(self, meta: CChanPlotMeta, pos: Tuple[int, int], x_limits, width=0.4):
        row, col = pos
        macd_lst = [
            [klu.macd.DIF, klu.macd.DEA, klu.macd.macd] for klu in meta.klu_iter()
        ]
        assert macd_lst[0] is not None, (
            "you can't draw macd until you delete macd_metric=False"
        )
        x_begin, x_end = x_limits[0], x_limits[1]
        macd_lst = np.array(macd_lst)
        x_idx = np.arange(macd_lst.shape[0])[x_begin:]
        dif_line = macd_lst[x_begin:x_end, 0]
        dea_line = macd_lst[x_begin:x_end, 1]
        macd_bar = macd_lst[x_begin:x_end, 2]

        y_min = np.min(macd_lst[x_begin:x_end, :])
        y_max = np.max(macd_lst[x_begin:x_end, :])

        # Draw DIF line
        self.figure.add_trace(
            go.Scatter(
                x=x_idx,
                y=dif_line,
                mode="lines",
                line=dict(color="#FFA500"),
                name="DIF",
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Draw DEA line
        self.figure.add_trace(
            go.Scatter(
                x=x_idx,
                y=dea_line,
                mode="lines",
                line=dict(color="#0000ff"),
                name="DEA",
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Draw MACD bars with color based on value
        # colors = ["#006400" if val < 0 else "red" for val in macd_bar]
        colors = np.array(
            ["red" if value > 0 else "#006400" for value in macd_bar], dtype=str
        )
        self.figure.add_trace(
            go.Bar(
                x=x_idx,
                y=macd_bar,
                marker_color=colors,
                width=width,
                name="MACD",
                text=self.text_vals_macd,
                hoverinfo="text",
            ),
            row=row,
            col=col,
        )

        # Update y-axis range
        self.figure.update_yaxes(range=[y_min, y_max], row=row, col=col)

    def draw_mean(self, meta: CChanPlotMeta, pos: Tuple[int, int]):
        row, col = pos
        mean_lst = [klu.trend[TREND_TYPE.MEAN] for klu in meta.klu_iter()]
        Ts = list(mean_lst[0].keys())

        # Create a color map similar to matplotlib's hsv
        # We'll create colors manually for Plotly
        colors = [f"hsl({360 * i / len(Ts)}, 100%, 50%)" for i in range(len(Ts))]

        for idx, T in enumerate(Ts):
            mean_arr = [mean_dict[T] for mean_dict in mean_lst]

            self.figure.add_trace(
                go.Scatter(
                    x=list(range(len(mean_arr))),
                    y=mean_arr,
                    mode="lines",
                    line=dict(color=colors[idx]),
                    name=f"{T} meanline",
                ),
                row=row,
                col=col,
            )

    def draw_channel(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        T=None,
        top_color="r",
        bottom_color="b",
        linewidth=3,
        linestyle="solid",
    ):
        row, col = pos
        max_lst = [klu.trend[TREND_TYPE.MAX] for klu in meta.klu_iter()]
        min_lst = [klu.trend[TREND_TYPE.MIN] for klu in meta.klu_iter()]
        config_T_lst = sorted(list(max_lst[0].keys()))

        if T is None:
            T = config_T_lst[-1]
        elif T not in max_lst[0]:
            raise CChanException(
                f"plot channel of T={T} is not setted in CChanConfig.trend_metrics = {config_T_lst}",
                ErrCode.PLOT_ERR,
            )

        top_array = [_d[T] for _d in max_lst]
        bottom_array = [_d[T] for _d in min_lst]

        # Convert matplotlib linestyle to plotly dash
        dash_style = "solid" if linestyle == "solid" else "dash"

        # Plot top channel
        self.figure.add_trace(
            go.Scatter(
                x=list(range(len(top_array))),
                y=top_array,
                mode="lines",
                line=dict(color=top_color, width=linewidth, dash=dash_style),
                name=f"{T}-TOP-channel",
            ),
            row=row,
            col=col,
        )

        # Plot bottom channel
        self.figure.add_trace(
            go.Scatter(
                x=list(range(len(bottom_array))),
                y=bottom_array,
                mode="lines",
                line=dict(color=bottom_color, width=linewidth, dash=dash_style),
                name=f"{T}-BOTTOM-channel",
            ),
            row=row,
            col=col,
        )

    def draw_boll(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        mid_color="black",
        up_color="blue",
        down_color="purple",
    ):
        row, col = pos
        x_begin = x_limits[0]

        try:
            ma = [klu.boll.MID for klu in meta.klu_iter()][x_begin:]
            up = [klu.boll.UP for klu in meta.klu_iter()][x_begin:]
            down = [klu.boll.DOWN for klu in meta.klu_iter()][x_begin:]
        except AttributeError as e:
            raise CChanException(
                "you can't draw boll until you set boll_n in CChanConfig",
                ErrCode.PLOT_ERR,
            ) from e

        x_range = list(range(x_begin, x_begin + len(ma)))

        # Plot mid line
        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=ma,
                mode="lines",
                line=dict(color=mid_color),
                name="BOLL MID",
            ),
            row=row,
            col=col,
        )

        # Plot upper line
        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=up,
                mode="lines",
                line=dict(color=up_color),
                name="BOLL UP",
            ),
            row=row,
            col=col,
        )

        # Plot lower line
        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=down,
                mode="lines",
                line=dict(color=down_color),
                name="BOLL DOWN",
            ),
            row=row,
            col=col,
        )

        # Update y-axis range
        self.y_min = min([self.y_min, min(down)])
        self.y_max = max([self.y_max, max(up)])

    def bsp_common_draw(
        self,
        bsp_list,
        pos: Tuple[int, int],
        x_limits,
        buy_color,
        sell_color,
        fontsize,
        arrow_l,
        arrow_h,
        arrow_w,
    ):
        row, col = pos
        x_begin = x_limits[0]
        y_range = self.y_max - self.y_min

        for bsp in bsp_list:
            if bsp.x < x_begin:
                continue

            color = buy_color if bsp.is_buy else sell_color
            arrow_dir = 1 if bsp.is_buy else -1
            arrow_len = arrow_l * y_range
            arrow_head = 1.0

            arrow_head = max(arrow_head, 0.3)  # 确保箭头尺寸至少为 0.3

            # Add text annotation
            self.figure.add_annotation(
                x=bsp.x,
                y=bsp.y - arrow_len * arrow_dir,
                text=f"{bsp.desc()}",
                font=dict(size=fontsize, color=color),
                showarrow=False,
                yanchor="top" if bsp.is_buy else "bottom",
                xanchor="center",
                row=row,
                col=col,
            )

            # Add arrow
            self.figure.add_annotation(
                x=bsp.x,
                y=bsp.y,
                xref="x",  # 明确 x 用数据坐标
                yref="y",  # 明确 y 用数据坐标
                ax=bsp.x,
                ay=bsp.y - arrow_len * arrow_dir,
                axref="x",  # 明确 ax 用数据坐标
                ayref="y",  # 明确 ay 用数据坐标
                showarrow=True,
                arrowhead=1,
                arrowwidth=arrow_w,
                arrowcolor=color,
                arrowsize=arrow_head,
                row=row,
                col=col,
            )

            # Update y range
            if bsp.y - arrow_len * arrow_dir < self.y_min:
                self.y_min = bsp.y - arrow_len * arrow_dir
            if bsp.y - arrow_len * arrow_dir > self.y_max:
                self.y_max = bsp.y - arrow_len * arrow_dir

    def draw_bs_point(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        buy_color="red",
        sell_color="green",
        fontsize=15,
        arrow_l=0.15,
        arrow_h=0.2,
        arrow_w=3,
    ):
        self.bsp_common_draw(
            bsp_list=meta.bs_point_lst,
            pos=pos,
            x_limits=x_limits,
            buy_color=buy_color,
            sell_color=sell_color,
            fontsize=fontsize,
            arrow_l=arrow_l,
            arrow_h=arrow_h,
            arrow_w=arrow_w,
        )

    def draw_seg_bs_point(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        buy_color="r",
        sell_color="g",
        fontsize=18,
        arrow_l=0.2,
        arrow_h=0.25,
        arrow_w=3,
    ):
        self.bsp_common_draw(
            bsp_list=meta.seg_bsp_lst,
            pos=pos,
            x_limits=x_limits,
            buy_color=buy_color,
            sell_color=sell_color,
            fontsize=fontsize,
            arrow_l=arrow_l,
            arrow_h=arrow_h,
            arrow_w=arrow_w,
        )

    def update_y_range(self, text_box, text_y):
        text_height = text_box.y1 - text_box.y0
        self.y_min = min([self.y_min, text_y - text_height])
        self.y_max = max([self.y_max, text_y + text_height])

    def plot_closeAction(
        self,
        plot_cover,
        cbsp,
        pos: Tuple[int, int],
        text_y,
        arrow_len,
        arrow_dir,
        color,
    ):
        row, col = pos
        if not plot_cover:
            return

        for closeAction in cbsp.close_action:
            self.figure.add_shape(
                type="line",
                x0=cbsp.x,
                y0=text_y,
                x1=closeAction.x,
                y1=text_y + arrow_len * arrow_dir + (closeAction.y - cbsp.y),
                line=dict(color=color, width=1),
                row=row,
                col=col,
            )

    def draw_marker(
        self,
        meta: CChanPlotMeta,
        pos: Tuple[int, int],
        x_limits,
        markers: Dict[
            CTime | str,
            Tuple[str, Literal["up", "down"], str] | Tuple[str, Literal["up", "down"]],
        ],
        arrow_l=0.15,
        arrow_h_r=0.2,
        arrow_w=1.0,
        fontsize=14,
        default_color="b",
    ):
        # {'2022/03/01': ('xxx', 'up', 'red'), '2022/03/02': ('yyy', 'down')}
        row, col = pos
        x_begin, x_end = x_limits
        datetick_dict = {date: idx for idx, date in enumerate(meta.datetick)}

        new_marker = {}
        for klu in meta.klu_iter():
            for date, marker in markers.items():
                date_str = date.to_str() if isinstance(date, CTime) else date
                if klu.include_sub_lv_time(date_str) and klu.time.to_str() != date_str:
                    new_marker[klu.time.to_str()] = marker
        new_marker.update(markers)

        kl_dict = dict(enumerate(meta.klu_iter()))
        y_range = self.y_max - self.y_min
        arrow_len = arrow_l * y_range
        arrow_h = arrow_len * arrow_h_r

        for date, marker in new_marker.items():
            if isinstance(date, CTime):
                date = date.to_str()
            if date not in datetick_dict:
                continue

            x = datetick_dict[date]
            if x < x_begin or x > x_end:
                continue

            if len(marker) == 2:
                color = default_color
                marker_content, position = marker
            else:
                assert len(marker) == 3
                marker_content, position, color = marker

            assert position in ["up", "down"]
            _dir = -1 if position == "up" else 1
            bench = kl_dict[x].high if position == "up" else kl_dict[x].low

            # Add arrow
            self.figure.add_annotation(
                x=x,
                y=bench - arrow_len * _dir,
                ax=0,
                ay=(arrow_len - arrow_h) * _dir,
                arrowhead=1,
                arrowwidth=arrow_w,
                arrowcolor=color,
                arrowsize=arrow_h,
                row=row,
                col=col,
            )

            # Add text
            self.figure.add_annotation(
                x=x,
                y=bench - arrow_len * _dir,
                text=marker_content,
                font=dict(size=fontsize, color=color),
                showarrow=False,
                yanchor="top" if position == "down" else "bottom",
                xanchor="center",
                row=row,
                col=col,
            )

    def draw_demark_begin_line(
        self,
        pos,
        begin_line_color,
        plot_begin_set: set,
        linestyle: str,
        demark_idx: T_DEMARK_INDEX,
    ):
        row, col = pos
        if (
            begin_line_color is not None
            and demark_idx["series"].TDST_peak is not None
            and id(demark_idx["series"]) not in plot_begin_set
        ):
            if demark_idx["series"].countdown is not None:
                end_idx = demark_idx["series"].countdown.kl_list[-1].idx
            else:
                end_idx = demark_idx["series"].kl_list[-1].idx

            # Convert matplotlib linestyle to plotly dash
            dash_style = "dash" if linestyle == "dashed" else "solid"

            self.figure.add_trace(
                go.Scatter(
                    x=[
                        demark_idx["series"].kl_list[CDemarkEngine.SETUP_BIAS].idx,
                        end_idx,
                    ],
                    y=[demark_idx["series"].TDST_peak, demark_idx["series"].TDST_peak],
                    mode="lines",
                    line=dict(color=begin_line_color, dash=dash_style),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )
            plot_begin_set.add(id(demark_idx["series"]))

    def draw_rsi(
        self,
        meta: CChanPlotMeta,
        pos,
        x_limits,
        color6="black",
        color12="orange",
        color24="red",
    ):
        if pos is None:
            return
        row, col = pos
        data = [[klu.rsi6, klu.rsi12, klu.rsi24] for klu in meta.klu_iter()]
        data = np.array(data)
        x_begin, x_end = x_limits
        x_range = np.arange(x_begin, x_end)
        data6 = data[x_begin:x_end, 0]
        data12 = data[x_begin:x_end, 1]
        data24 = data[x_begin:x_end, 2]

        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=data6,
                mode="lines",
                name="RSI6",
                line=dict(color=color6),
                showlegend=True,
                text=self.text_vals_rsi,
                hoverinfo="text",
            ),
            row=row,
            col=col,
        )

        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=data12,
                mode="lines",
                name="RSI12",
                line=dict(color=color12),
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=data24,
                mode="lines",
                name="RSI24",
                line=dict(color=color24),
                showlegend=True,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=[70] * len(x_range),
                mode="lines",
                line=dict(color="red"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=[30] * len(x_range),
                mode="lines",
                line=dict(color="green"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

    def draw_kdj(
        self,
        meta: CChanPlotMeta,
        pos,
        x_limits,
        k_color="orange",
        d_color="blue",
        j_color="pink",
    ):
        if pos is None:
            return
        row, col = pos
        kdj = [klu.kdj for klu in meta.klu_iter()]
        x_begin, x_end = x_limits
        x_range = list(range(x_begin, x_end))

        # Add K line
        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=[x.k for x in kdj][x_begin:x_end],
                mode="lines",
                line=dict(color=k_color),
                name="K",
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Add D line
        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=[x.d for x in kdj][x_begin:x_end],
                mode="lines",
                line=dict(color=d_color),
                name="D",
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        # Add J line
        self.figure.add_trace(
            go.Scatter(
                x=x_range,
                y=[x.j for x in kdj][x_begin:x_end],
                mode="lines",
                line=dict(color=j_color),
                name="J",
                hoverinfo="text",
                text=self.text_vals_kdj,
            ),
            row=row,
            col=col,
        )

        # In plotly, legends are added by default when 'name' is specified

    def draw_demark(
        self,
        meta: CChanPlotMeta,
        pos,
        x_limits,
        setup_color="b",
        countdown_color="r",
        fontsize=12,
        min_setup=9,
        max_countdown_background="yellow",
        begin_line_color: Optional[str] = "purple",
        begin_line_style="dashed",
    ):  # sourcery skip: low-code-quality
        row, col = pos
        x_begin = x_limits[0]
        text_height: Optional[float] = None

        for klu in meta.klu_iter():
            if klu.idx < x_begin:
                continue
            under_bias, upper_bias = 0, 0
            plot_begin_set = set()

            # Process setup indicators
            for demark_idx in klu.demark.get_setup():
                if (
                    demark_idx["series"].idx < min_setup
                    or not demark_idx["series"].setup_finished
                ):
                    continue

                self.draw_demark_begin_line(
                    pos, begin_line_color, plot_begin_set, begin_line_style, demark_idx
                )

                # Add text annotation for setup
                y_position = (
                    klu.low - under_bias
                    if demark_idx["dir"] == BI_DIR.DOWN
                    else klu.high + upper_bias
                )
                y_anchor = "top" if demark_idx["dir"] == BI_DIR.DOWN else "bottom"

                annotation_props = {
                    "x": klu.idx,
                    "y": y_position,
                    "text": str(demark_idx["idx"]),
                    "font": {"size": fontsize, "color": setup_color},
                    "showarrow": False,
                    "yanchor": y_anchor,
                    "xanchor": "center",
                }

                self.figure.add_annotation(**annotation_props, row=row, col=col)

                # Use the getTextBox helper to estimate text height
                text_box = getTextBox(self.figure, annotation_props)
                if demark_idx["dir"] == BI_DIR.DOWN:
                    under_bias += text_box.height
                else:
                    upper_bias += text_box.height

            # Process countdown indicators
            for demark_idx in klu.demark.get_countdown():
                # Calculate box_bias
                if text_height is None:
                    # Initialize text_height with approximation from getTextBox
                    temp_annotation = {"text": "0", "font": {"size": fontsize}}
                    text_height = getTextBox(self.figure, temp_annotation).height

                box_bias = (
                    0.5 * text_height
                    if demark_idx["idx"] == CDemarkEngine.MAX_COUNTDOWN
                    else 0
                )
                y_position = (
                    klu.low - under_bias - box_bias
                    if demark_idx["dir"] == BI_DIR.DOWN
                    else klu.high + upper_bias + box_bias
                )
                y_anchor = "top" if demark_idx["dir"] == BI_DIR.DOWN else "bottom"

                # Create annotation properties for countdown
                annotation_props = {
                    "x": klu.idx,
                    "y": y_position,
                    "text": str(demark_idx["idx"]),
                    "font": {"size": fontsize, "color": countdown_color},
                    "showarrow": False,
                    "yanchor": y_anchor,
                    "xanchor": "center",
                }

                # Add background for max countdown
                if demark_idx["idx"] == CDemarkEngine.MAX_COUNTDOWN:
                    annotation_props.update(
                        {
                            "bgcolor": max_countdown_background,
                            "bordercolor": max_countdown_background,
                            "borderpad": 0,
                        }
                    )

                self.figure.add_annotation(**annotation_props, row=row, col=col)

                # Use getTextBox helper to calculate text dimensions
                text_box = getTextBox(self.figure, annotation_props)
                if demark_idx["dir"] == BI_DIR.DOWN:
                    under_bias += text_box.height
                else:
                    upper_bias += text_box.height


# Helper function to approximate text box dimensions in plotly (since there's no direct equivalent)
def getTextBox(fig, annotation_props):
    # This is an approximation since plotly doesn't have the same text box measurement
    # We estimate the size based on text length and font size
    font_size = annotation_props.get("font", {}).get("size", 12)
    text = annotation_props.get("text", "")
    text_len = len(str(text))

    # Very rough estimation of text dimensions
    width = text_len * font_size * 0.6
    height = font_size * 1.2

    # Return an object with width and height properties
    class TextBox:
        def __init__(self, w, h):
            self.width = w
            self.height = h

    return TextBox(width, height)


def plot_bi_element(bi: CBi_meta, fig, pos, color: str):
    row, col = pos
    if bi.is_sure:
        dash = "solid"
    else:
        dash = "dash"

    fig.add_trace(
        go.Scatter(
            x=[bi.begin_x, bi.end_x],
            y=[bi.begin_y, bi.end_y],
            mode="lines",
            line=dict(color=color, dash=dash),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )


def bi_text(bi_idx, fig, bi, end_fontsize, end_color, pos):
    row, col = pos

    if bi_idx == 0:
        # Add annotation for beginning point of first bi
        fig.add_annotation(
            x=bi.begin_x,
            y=bi.begin_y,
            text=f"{bi.begin_y:.5f}",
            font=dict(size=end_fontsize, color=end_color),
            showarrow=False,
            yanchor="top" if bi.dir == BI_DIR.UP else "bottom",
            xanchor="center",
            row=row,
            col=col,
        )

    # Add annotation for end point
    fig.add_annotation(
        x=bi.end_x,
        y=bi.end_y,
        text=f"{bi.end_y:.5f}",
        font=dict(size=end_fontsize, color=end_color),
        showarrow=False,
        yanchor="top" if bi.dir == BI_DIR.DOWN else "bottom",
        xanchor="center",
        row=row,
        col=col,
    )


def show_func_helper(func):
    print(f"{func.__name__}:")
    insp = inspect.signature(func)
    for name, para in insp.parameters.items():
        if para.default == inspect.Parameter.empty:
            continue
            # print(f"\t{name}*")
        elif isinstance(para.default, str):
            print(f"\t{name}: '{para.default}'")
        else:
            print(f"\t{name}: {para.default}")


def add_zs_text(fig, zs_meta: CZS_meta, fontsize, text_color, pos):
    row, col = pos

    # Add lower bound text
    fig.add_annotation(
        x=zs_meta.begin,
        y=zs_meta.low,
        text=f"{zs_meta.low:.2f}",
        font=dict(size=fontsize, color=text_color),
        showarrow=False,
        yanchor="top",
        xanchor="center",
        row=row,
        col=col,
    )

    # Add upper bound text
    fig.add_annotation(
        x=zs_meta.begin + zs_meta.w,
        y=zs_meta.low + zs_meta.h,
        text=f"{zs_meta.low + zs_meta.h:.2f}",
        font=dict(size=fontsize, color=text_color),
        showarrow=False,
        yanchor="bottom",
        xanchor="center",
        row=row,
        col=col,
    )
