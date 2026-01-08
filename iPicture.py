import streamlit as st
import numpy as np
import cv2
import pandas as pd

# -------------------- 页面配置 --------------------
st.set_page_config(page_title="欢迎使用 iPicture", page_icon="🖼️", layout="centered")

# -------------------- 主页标题 --------------------
st.markdown("<h1 style='text-align:center;'>🖼️ iPicture (V1.0)</h1>", unsafe_allow_html=True)

# 设置登录引导
c1, c2 = st.columns([1, 2])
with c2:
    st.write(f'<span style="color: blue; font-size: 20px; font-weight: bold;">'
            f'<br>欢迎使用本工具，本工具用于生成测试图片。'
            f'<br>👈请先在左侧登录后，再使用'
            , unsafe_allow_html=True)

# -------------------- 用户库 --------------------
USER_DB = {"cyan": "Cyan@123", "vrpnl": "vrpnl123"}

# -------------------- session_state 初始化 --------------------
for key, val in {"logged_in": False, "username": "", "img": None}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# -------------------- 侧边栏登录 --------------------
with st.sidebar:
    if st.session_state.logged_in:
        st.success(f"欢迎回来，**{st.session_state.username}**！")
        if st.button("退出登录"):
            for k in ["logged_in", "username", "img"]:
                st.session_state[k] = "" if k == "username" else None
            st.rerun()
    else:
        st.title("请登录")
        user = st.text_input("用户名")
        pwd = st.text_input("密码", type="password")
        if st.button("登录"):
            if USER_DB.get(user) == pwd:
                st.session_state.logged_in = True
                st.session_state.username = user
                st.rerun()
            else:
                st.error("用户名或密码错误！")

# -------------------- 生成函数合集 --------------------
def gen_solid(h, w, b, g, r):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = b
    img[:, :, 1] = g
    img[:, :, 2] = r
    return img


def gen_flk_zl(h, w, gray):
    """紫绿竖条纹，周期 20 px（可改）"""
    stripe_w = 1
    pattern = (np.arange(w) // stripe_w) % 2          # 0 1 0 1 ...
    img = np.zeros((h, w, 3), np.uint8)
    img[:, pattern == 0] = (gray, 0, gray)   # 紫  B=gray,G=0,R=gray
    img[:, pattern == 1] = (0, gray, 0)      # 绿  B=0,G=gray,R=0
    return img


def gen_flk_hh(h, w, gray):
    """黑灰竖条纹，周期 20 px（可改）"""
    stripe_w = 1
    pattern = (np.arange(w) // stripe_w) % 2
    img = np.full((h, w, 3), gray, np.uint8)
    img[:, pattern == 0] = 0                 # 黑
    # pattern == 1 保持 gray                 # 灰
    return img


def gen_crosstalk(h, w, gray, mode="上下"):
    img = np.full((h, w, 3), gray, dtype=np.uint8)
    dh, dw = h // 3, w // 3          # 高、宽各 1/3
    white = (255, 255, 255)

    if mode == "上下":
        x1 = x2 = (w - dw) // 2;  y1 = 0;       y2 = h - dh
    elif mode == "左右":
        y1 = y2 = (h - dh) // 2;  x1 = 0;       x2 = w - dw
    elif mode == "上连续":
        x1 = x2 = (w - dw) // 2;  y1 = 0;       y2 = dh
    elif mode == "下连续":
        x1 = x2 = (w - dw) // 2;  y1 = h - 2*dh; y2 = h - dh
    elif mode == "左连续":
        y1 = y2 = (h - dh) // 2;  x1 = 0;       x2 = dw
    elif mode == "右连续":
        y1 = y2 = (h - dh) // 2;  x1 = w - 2*dw; x2 = w - dw
    else:
        raise ValueError("未知模式")

    cv2.rectangle(img, (x1, y1), (x1 + dw, y1 + dh), white, -1)
    cv2.rectangle(img, (x2, y2), (x2 + dw, y2 + dh), white, -1)
    return img

COLOR_MAP = {
    "红": (0, 0, 255),
    "绿": (0, 255, 0),
    "蓝": (255, 0, 0),
    "白": (255, 255, 255),
    "黑": (0, 0, 0)
}

def gen_marker(h, w, bg_gray, row_pos=None, col_pos=None, color_name="红", line_thk=1):
    img = np.full((h, w, 3), bg_gray, dtype=np.uint8)
    color = COLOR_MAP[color_name]

    if row_pos is not None and 0 <= row_pos < h:
        cv2.line(img, (0, row_pos), (w, row_pos), color, line_thk)
    if col_pos is not None and 0 <= col_pos < w:
        cv2.line(img, (col_pos, 0), (col_pos, h), color, line_thk)
    return img


def gen_custom_periodic(h: int, w: int, ph: int, pw: int, df: "pd.DataFrame") -> np.ndarray:
    """
    将用户编辑的周期表格平铺成完整图像
    h, w   : 目标图像高宽
    ph, pw : 周期高宽
    df     : ph×pw DataFrame，每个单元格是 "R,G,B" 字符串
    return : (h, w, 3) 的 uint8 阵列
    """
    pattern = np.zeros((ph, pw, 3), dtype=np.uint8)
    for r in range(ph):
        for c in range(pw):
            try:
                r_str, g_str, b_str = edited_df.iloc[r, c].split(',')
                r_val, g_val, b_val = map(int, (r_str, g_str, b_str))
                pattern[r, c] = np.clip([b_val, g_val, r_val], 0, 255)  # RGB→BGR
            except Exception:
                pattern[r, c] = 0

    # 平铺
    tile_r = (h + ph - 1) // ph
    tile_c = (w + pw - 1) // pw
    big = np.tile(pattern, (tile_r, tile_c, 1))[:h, :w, :]
    return big


# -------------------- 主界面（登录后） --------------------
if st.session_state.logged_in:
    # st.subheader("步骤一：设置分辨率")
    st.write(f'<span style="font-size: 16px; font-weight: bold;">步骤一：请设置图片分辨率'
             , unsafe_allow_html=True)
    c21, c22, c23, c24, c25 = st.columns([1, 5, 1, 5, 10])
    with c22:
        rows = st.number_input("行数 (高)", min_value=1, value=480)
    with c24:
        cols = st.number_input("列数 (宽)", min_value=1, value=640)

    # st.subheader("步骤二：选择图片类型")
    st.write(f'<span style="font-size: 16px; font-weight: bold;">步骤二：请选择图片类型'
             , unsafe_allow_html=True)
    c31, c32, c33, c34, c35, c36, c37 = st.columns([1, 5, 1, 5, 1, 5, 1])
    with c32:
        img_type = st.selectbox(
            "图片类型",
            ["solid图片", "flk紫绿图片", "flk黑灰图片", "R1 crosstalk图片", "标记线图片", "自定义周期"]
        )

    # 动态参数区
    # st.subheader("步骤三：补充参数")
    st.write(f'<span style="font-size: 16px; font-weight: bold;">步骤三：补充参数并点击按钮生成图片'
             , unsafe_allow_html=True)
    if img_type == "solid图片":
        c41, c42, c43, c44, c45, c46, c47 = st.columns([1, 5, 1, 5, 1, 5, 1])
        r = c42.number_input("R通道", 0, 255, 128)
        g = c44.number_input("G通道", 0, 255, 128)
        b = c46.number_input("B通道", 0, 255, 128)
        args = (r, g, b)
        func = gen_solid
    elif img_type == "flk紫绿图片":
        c41, c42, c43, c44, c45, c46, c47 = st.columns([1, 5, 1, 5, 1, 5, 1])
        gray = c42.number_input("紫绿灰阶", 0, 255, 128)
        args = (gray,)
        func = gen_flk_zl
    elif img_type == "flk黑灰图片":
        c41, c42, c43, c44, c45, c46, c47 = st.columns([1, 5, 1, 5, 1, 5, 1])
        gray = c42.number_input("黑灰灰阶", 0, 255, 128)
        args = (gray,)
        func = gen_flk_hh
    elif img_type == "R1 crosstalk图片":
        c41, c42, c43, c44, c45, c46, c47 = st.columns([1, 5, 1, 5, 1, 5, 1])
        gray = c42.number_input("背景灰阶", 0, 255, 64)
        mode = c44.selectbox("白块布局",
                            ["上下", "左右", "上连续", "下连续", "左连续", "右连续"])
        args = (gray, mode)
        func = gen_crosstalk
    elif img_type == "标记线图片":
        c41, c42, c43, c44, c45, c46, c47 = st.columns([1, 5, 1, 5, 1, 5, 1])
        bg    = c42.number_input("背景灰阶", 0, 255, 128)
        r_pos = c44.number_input("行位置(-1为没有)", -1, rows-1, rows//2)
        c_pos = c46.number_input("列位置(-1为没有)", -1, cols-1, cols//2)

        line_color = c42.selectbox("线颜色", ["红", "绿", "蓝", "白", "黑"])
        line_thickness  = c44.number_input("线宽", 1, 20, 1)

        args = (bg,
                r_pos if r_pos != -1 else None,
                c_pos if c_pos != -1 else None,
                line_color,
                line_thickness)
        func = gen_marker

    elif img_type == "自定义周期":
        c41, c42, c43, c44, c45, c46, c47 = st.columns([1, 5, 1, 5, 1, 5, 1])
        ph = c42.number_input("周期行数", 1, rows, min(4, rows), key="ph")
        pw = c44.number_input("周期列数", 1, cols, min(4, cols), key="pw")
        c87, c88, c89 = st.columns([1, 15, 1])
        with c88:
            df = pd.DataFrame("128,128,128", index=range(ph), columns=range(pw))
            edited_df = st.data_editor(df, use_container_width=True)

        args = (ph,
                pw,
                edited_df)
        func = gen_custom_periodic
    c41, c42, c43, c44, c45, c46, c47 = st.columns([1, 5, 1, 5, 1, 5, 1])
    if c42.button("生成图片"):
        st.session_state.img = func(rows, cols, *args)

    # 显示结果
    if st.session_state.img is not None:
        c41, c42, c43, c44, c45, c46, c47 = st.columns([1, 5, 1, 5, 1, 5, 1])

        with  c42:
            if st.download_button(
                label="下载图片",
                data=cv2.imencode(".bmp", st.session_state.img)[1].tobytes(),
                file_name=f"{img_type}.bmp",
                mime="image/bmp"
                ):
                st.balloons()
        c51, c52, c53 = st.columns([1, 15, 1])
        with c52:
            st.image(st.session_state.img, channels="BGR", use_container_width=True)
else:
    st.warning("请先登录后再使用生成工具！")

# 编辑上传按钮底色
st.markdown(
    '''
    <style>
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1w723zb.e4man114 > div > div:nth-child(10) > div > div:nth-child(2) > div > div > div > button
    {
        background-color: rgb(220, 240, 240);
        height: 60px;
        width: 120px;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1w723zb.e4man114 > div > div:nth-child(11) > div > div:nth-child(2) > div > div > div > button
    {
        background-color: rgb(220, 240, 240);
        height: 60px;
        width: 120px;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-1w723zb.e4man114 > div > div:nth-child(9) > div > div:nth-child(2) > div > div > div > button
    {
        background-color: rgb(220, 240, 220);
        height: 60px;
        width: 120px;
    }
    #root > div:nth-child(1) > div.withScreencast > div > div > section > div.st-emotion-cache-155jwzh.e6f82ta2 > div.st-emotion-cache-1r1cntt.e6f82ta1 > div > div > div.stElementContainer.element-container.st-emotion-cache-zh2fnc.e1wguzas1 > div > button
    {
        background-color: rgb(220, 240, 220);
    }
    </style>
    ''',
    unsafe_allow_html=True
)
