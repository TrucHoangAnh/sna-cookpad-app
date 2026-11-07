"""
app.py - Ứng dụng demo: khám phá mạng đồng xuất hiện nguyên liệu trong công thức Cookpad Việt Nam.

Cách chạy (trong thư mục chứa app.py, sna_core.py và thư mục cookpad_data):
    streamlit run app.py
"""
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

import sna_core as core

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
           "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

st.set_page_config(page_title="Mạng nguyên liệu Cookpad", layout="wide")
st.title("Mạng đồng xuất hiện nguyên liệu trong công thức Việt Nam")
st.caption(
    "Nguồn: Cookpad Việt Nam. Nút là nguyên liệu. Cạnh nối hai nguyên liệu "
    "cùng xuất hiện trong một công thức, độ đậm theo trọng số NPMI."
)

with st.sidebar:
    st.header("Nguồn dữ liệu")
    edges_path = st.text_input("File cạnh", "cookpad_data/graph_v5/edges.csv")
    nodes_path = st.text_input("File node và chỉ số", "cookpad_data/analysis_v5/gephi_nodes.csv")
    rec_path = st.text_input("File công thức sạch (tuỳ chọn)", "cookpad_data/clean_v5/recipe_ingredients.csv")
    st.caption("Đường dẫn tương đối tính từ thư mục chạy lệnh streamlit.")


@st.cache_resource(show_spinner="Đang nạp mạng...")
def get_data(edges_file, nodes_file, rec_file):
    G, nodes = core.load_network(edges_file, nodes_file)
    rec = core.load_recipes(rec_file)
    return G, nodes, rec


try:
    G, nodes, rec = get_data(edges_path, nodes_path, rec_path)
except (FileNotFoundError, KeyError, ValueError) as ex:
    st.error(f"Không nạp được dữ liệu: {ex}. Kiểm tra lại đường dẫn ở thanh bên.")
    st.stop()


def plot_ego(center, top_k):
    """Vẽ nguyên liệu trung tâm và các láng giềng mạnh nhất."""
    members = core.ego_nodes(G, center, top_k)
    H = G.subgraph(members)
    pos = nx.spring_layout(H, weight="weight", seed=42)
    max_strength = max(float(H.nodes[n]["strength"]) for n in H.nodes) or 1.0

    edge_x, edge_y = [], []
    for a, b in H.edges():
        edge_x += [pos[a][0], pos[b][0], None]
        edge_y += [pos[a][1], pos[b][1], None]

    node_x = [pos[n][0] for n in H.nodes]
    node_y = [pos[n][1] for n in H.nodes]
    sizes = []
    colors = []
    hover = []
    for n in H.nodes:
        d = H.nodes[n]
        size = 12 + 36 * float(d["strength"]) / max_strength
        sizes.append(size * (1.4 if n == center else 1.0))
        colors.append(PALETTE[int(d["community"]) % len(PALETTE)])
        hover.append(f"{n}<br>số công thức: {int(d['recipes'])}<br>cộng đồng: {int(d['community'])}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1, color="#bbbbbb"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             text=list(H.nodes), textposition="top center",
                             hovertext=hover, hoverinfo="text",
                             marker=dict(size=sizes, color=colors,
                                         line=dict(width=1, color="white"))))
    fig.update_layout(showlegend=False, height=520, plot_bgcolor="white",
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=0, r=0, t=10, b=0))
    return fig


names = sorted(nodes["Id"].tolist())
top_default = nodes.sort_values("strength", ascending=False)["Id"].iloc[0]

tab_overview, tab_lookup, tab_rank, tab_comm, tab_suggest = st.tabs(
    ["Tổng quan", "Tra cứu nguyên liệu", "Xếp hạng", "Cộng đồng", "Gợi ý món"]
)

# ---------------------------------------------------------------- Tổng quan
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nguyên liệu", G.number_of_nodes())
    c2.metric("Quan hệ", G.number_of_edges())
    c3.metric("Cộng đồng", int(nodes["community"].nunique()))
    c4.metric("Mật độ", f"{nx.density(G):.3f}")

    st.subheader("Các cộng đồng")
    st.dataframe(core.community_table(nodes), hide_index=True)

    st.info(
        "**Cách đọc:** nguyên liệu càng nhiều quan hệ mạnh thì càng ở trung tâm. "
        "Nguyên liệu có betweenness cao là cầu nối giữa các nhóm món. "
        "Cộng đồng là nhóm nguyên liệu hay xuất hiện cùng nhau."
    )

# ---------------------------------------------------------------- Tra cứu
with tab_lookup:
    name = st.selectbox("Chọn nguyên liệu", names, index=names.index(top_default))
    row = nodes.set_index("Id").loc[name]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số công thức", int(row["recipes"]))
    c2.metric("Bậc", int(row["degree"]))
    c3.metric("Betweenness", f"{float(row['betweenness']):.4f}")
    c4.metric("Cộng đồng", int(row["community"]))
    if int(row["is_staple"]) == 1:
        st.caption("Đây là gia vị nền, nên thường nối với hầu hết các món.")

    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Nguyên liệu đi cùng mạnh nhất**")
        st.dataframe(core.neighbors(G, name, 15), hide_index=True)
    with right:
        top_k = st.slider("Số láng giềng hiển thị", 5, 40, 25)
        st.plotly_chart(plot_ego(name, top_k))

# ---------------------------------------------------------------- Xếp hạng
with tab_rank:
    measure = st.selectbox("Độ đo", list(core.MEASURES),
                           format_func=lambda k: core.MEASURES[k])
    n_top = st.slider("Số lượng", 5, 50, 20)
    hide = st.checkbox("Ẩn gia vị nền", value=False)

    if measure == "closeness":
        st.warning("Closeness không ổn định giữa các phiên bản mạng, chỉ dùng để tham khảo.")

    df_rank = core.top_by_measure(nodes, measure, n_top, hide)
    fig = go.Figure(go.Bar(x=df_rank["giá trị"], y=df_rank["nguyên liệu"],
                           orientation="h", marker_color="#1f77b4"))
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      height=max(350, 24 * len(df_rank)),
                      margin=dict(l=0, r=10, t=10, b=10))
    st.plotly_chart(fig)
    st.dataframe(df_rank, hide_index=True)

# ---------------------------------------------------------------- Cộng đồng
with tab_comm:
    sizes = nodes["community"].value_counts()
    cid = st.selectbox(
        "Chọn cộng đồng",
        sizes.index.tolist(),
        format_func=lambda c: core.community_label(c, int(sizes[c])),
    )
    st.dataframe(core.community_members(nodes, cid), hide_index=True)
    st.caption(
        "Tên gợi ý theo phiên bản v5. Nếu bạn chạy lại phân tích và số thứ tự cộng đồng thay đổi, "
        "cần đặt lại tên trong sna_core.py."
    )

# ---------------------------------------------------------------- Gợi ý món
with tab_suggest:
    chosen = st.multiselect("Chọn nguyên liệu bạn đang có", names, default=[])
    if not chosen:
        st.info("Chọn một vài nguyên liệu để xem gợi ý nguyên liệu nên thêm và công thức gần nhất.")
    else:
        hide_s = st.checkbox("Bỏ gia vị nền khỏi gợi ý", value=True)

        st.subheader("Nguyên liệu nên thêm")
        sug = core.suggest(G, chosen, n=10, hide_staples=hide_s)
        if sug.empty:
            st.write("Không có gợi ý.")
        else:
            st.dataframe(sug, hide_index=True)
            st.caption(
                "Điểm NPMI là tổng độ mạnh kết nối với các nguyên liệu đã chọn. "
                "Điểm Adamic-Adar đo số láng giềng chung, nên dùng cùng điểm NPMI để đánh giá."
            )

        st.subheader("Công thức gần nhất")
        if rec is None:
            st.info("Chưa có file công thức sạch nên không tìm được món. "
                    "Kiểm tra đường dẫn file công thức ở thanh bên.")
        else:
            matches = core.matching_recipes(rec, chosen, 10)
            if matches.empty:
                st.write("Không có công thức nào khớp.")
            else:
                st.dataframe(matches, hide_index=True)
