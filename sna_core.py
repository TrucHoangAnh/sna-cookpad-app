"""
sna_core.py - Lõi tính toán cho ứng dụng demo mạng nguyên liệu.

File này không phụ thuộc Streamlit hay Plotly, nên có thể kiểm thử riêng.
Dữ liệu đầu vào lấy từ pipeline:
    cookpad_data/graph_v5/edges.csv              (từ build_graph.py)
    cookpad_data/analysis_v5/gephi_nodes.csv     (từ analyze_graph.py)
    cookpad_data/clean_v5/recipe_ingredients.csv (từ clean_ingredients.py, tuỳ chọn)
"""
import os

import networkx as nx
import pandas as pd

MEASURES = {
    "degree": "Bậc (degree): số đối tác khác nhau",
    "strength": "Sức mạnh (strength): tổng NPMI với các đối tác",
    "betweenness": "Betweenness: vai trò cầu nối giữa các nhóm",
    "eigenvector": "Eigenvector: kết nối với các nguyên liệu quan trọng",
    "pagerank": "PageRank: mức quan trọng theo trọng số NPMI",
    "closeness": "Closeness (chỉ tham khảo, độ ổn định thấp)",
}

# Tên gợi ý cho cộng đồng Louvain của phiên bản v5 (mục 9.3 trong tài liệu).
# Chỉ đúng khi số thứ tự cộng đồng không đổi; nếu chạy lại và thứ tự thay đổi thì cần đặt lại.
COMMUNITY_NAMES_V5 = {
    1: "Gia vị nền",
    2: "Sốt, nước chấm, bánh",
    3: "Rau thơm, sả, món chay",
    4: "Củ quả, canh, hầm",
    5: "Xào, nấm, tôm",
    6: "Bún, rau sống, gỏi",
    7: "Chè, nước cốt dừa, nước dùng",
    8: "Gừng, quế, hồi, thảo quả",
}


def load_network(edges_path, nodes_path):
    """Nạp mạng từ edges.csv và bảng node. Trả về (G, nodes_df)."""
    nodes = pd.read_csv(nodes_path, encoding="utf-8-sig")
    nodes["Id"] = nodes["Id"].astype(str)
    nodes = nodes.reset_index(drop=True)

    edges = pd.read_csv(edges_path, encoding="utf-8-sig")

    G = nx.Graph()
    for row in nodes.to_dict("records"):
        nid = row.pop("Id")
        G.add_node(nid, **row)
    for row in edges.itertuples(index=False):
        a, b = str(row.Source), str(row.Target)
        if a in G and b in G and a != b:
            G.add_edge(a, b, weight=float(row.Weight), cooc=int(row.cooc))
    return G, nodes


def load_recipes(path):
    """Nạp công thức sạch để tìm món. Trả về None nếu không có file."""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    return df[["recipe_id", "recipe_name", "ingredient"]].dropna()


def community_label(cid, size):
    name = COMMUNITY_NAMES_V5.get(int(cid))
    label = f"Cộng đồng {int(cid)}"
    if name:
        label += f" – {name}"
    return f"{label} ({size} nguyên liệu)"


def neighbors(G, name, top=15):
    """Các nguyên liệu đi cùng mạnh nhất với `name`, xếp theo NPMI."""
    rows = [(v, d["weight"], d["cooc"]) for v, d in G[name].items()]
    rows.sort(key=lambda r: -r[1])
    return pd.DataFrame(rows[:top], columns=["nguyên liệu kết hợp", "NPMI", "số công thức chung"]).round(3)


def top_by_measure(nodes, measure, n=20, hide_staples=False):
    """Top n nguyên liệu theo một độ đo centrality."""
    df = nodes[nodes["is_staple"] == 0] if hide_staples else nodes
    df = df.sort_values(measure, ascending=False).head(n)
    out = df[["Id", "recipes", measure, "community"]].copy()
    out.columns = ["nguyên liệu", "số công thức", "giá trị", "cộng đồng"]
    out["giá trị"] = out["giá trị"].round(4)
    return out.reset_index(drop=True)


def community_table(nodes, top_k=6):
    """Bảng tóm tắt mỗi cộng đồng: kích thước và các nguyên liệu tiêu biểu."""
    rows = []
    for cid, grp in nodes.sort_values("strength", ascending=False).groupby("community"):
        rows.append({
            "cộng đồng": int(cid),
            "số nguyên liệu": len(grp),
            "tên gợi ý": COMMUNITY_NAMES_V5.get(int(cid), ""),
            "tiêu biểu": ", ".join(grp["Id"].head(top_k)),
        })
    return pd.DataFrame(rows).sort_values("số nguyên liệu", ascending=False).reset_index(drop=True)


def community_members(nodes, cid):
    df = nodes[nodes["community"] == cid].sort_values("strength", ascending=False)
    out = df[["Id", "recipes", "degree", "strength", "is_staple"]].copy()
    out.columns = ["nguyên liệu", "số công thức", "bậc", "sức mạnh", "gia vị nền"]
    out["sức mạnh"] = out["sức mạnh"].round(3)
    return out.reset_index(drop=True)


def ego_nodes(G, center, top_k=25):
    """Nguyên liệu trung tâm cùng top_k láng giềng mạnh nhất (để vẽ)."""
    nbrs = sorted(G[center].items(), key=lambda kv: -kv[1]["weight"])[:top_k]
    return [center] + [v for v, _ in nbrs]


def suggest(G, selected, n=10, hide_staples=True):
    """
    Gợi ý nguyên liệu nên thêm, dựa trên các nguyên liệu đã chọn.
    - điểm NPMI: tổng NPMI giữa ứng viên và các nguyên liệu đã chọn.
    - điểm Adamic-Adar: tổng chỉ số Adamic-Adar (số láng giềng chung, có hiệu chỉnh bậc).
    """
    sel = [s for s in selected if s in G]
    if not sel:
        return pd.DataFrame()

    cand = {}
    for s in sel:
        for c, d in G[s].items():
            if c in sel:
                continue
            if hide_staples and G.nodes[c].get("is_staple"):
                continue
            info = cand.setdefault(c, {"npmi": 0.0, "with": []})
            info["npmi"] += d["weight"]
            info["with"].append(s)
    if not cand:
        return pd.DataFrame()

    pairs = [(c, s) for c in cand for s in sel]
    aa = {}
    for u, v, p in nx.adamic_adar_index(G, pairs):
        aa[(u, v)] = p

    rows = []
    for c, info in cand.items():
        rows.append({
            "gợi ý": c,
            "điểm NPMI": round(info["npmi"], 3),
            "điểm Adamic-Adar": round(sum(aa.get((c, s), 0.0) for s in sel), 3),
            "đi cùng": ", ".join(info["with"]),
        })
    out = pd.DataFrame(rows).sort_values(["điểm NPMI", "điểm Adamic-Adar"], ascending=False)
    return out.head(n).reset_index(drop=True)


def matching_recipes(rec, selected, n=10):
    """Công thức dùng được nhiều nguyên liệu đã chọn nhất (ưu tiên tỷ lệ khớp cao)."""
    if rec is None or not selected:
        return pd.DataFrame()
    sel = set(selected)

    total = rec.groupby("recipe_id").size()
    need = rec.groupby("recipe_id")["ingredient"].agg(set)

    hit = (rec[rec["ingredient"].isin(sel)]
           .groupby(["recipe_id", "recipe_name"])["ingredient"]
           .agg(lambda s: sorted(set(s)))
           .reset_index(name="đã có"))
    if hit.empty:
        return pd.DataFrame()

    hit["số khớp"] = hit["đã có"].str.len()
    hit["tổng nguyên liệu"] = hit["recipe_id"].map(total)
    hit["tỷ lệ khớp"] = (hit["số khớp"] / hit["tổng nguyên liệu"]).round(2)
    hit = hit.sort_values(["tỷ lệ khớp", "số khớp"], ascending=False).head(n)

    hit["còn thiếu (tối đa 6)"] = [
        ", ".join(sorted(need[r] - sel)[:6]) for r in hit["recipe_id"]
    ]
    hit["đã có"] = hit["đã có"].apply(", ".join)
    return hit[["recipe_name", "số khớp", "tổng nguyên liệu", "tỷ lệ khớp",
                "đã có", "còn thiếu (tối đa 6)"]].rename(columns={"recipe_name": "tên món"}).reset_index(drop=True)
