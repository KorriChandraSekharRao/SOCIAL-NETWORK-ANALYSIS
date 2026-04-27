import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import gradio as gr
from collections import Counter, defaultdict
from networkx.algorithms.community.quality import modularity
import io
import base64

# ---------------- GLOBAL ----------------
G_real = None
G_syn = None
G_event = None
G_places = None

# ---------------- HELPER: FIGURE TO IMAGE ----------------
def fig_to_img(fig):
    """Convert matplotlib figure to image for Gradio"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close(fig)
    return buf

# ---------------- LOAD SURVEY DATA ----------------
def load_survey(file):
    global G_real
    
    if file is None:
        return None, "Please upload an Excel file.", ""
    
    df = pd.read_excel(file.name)
    G = nx.Graph()

    # STEP 1: ADD PRIMARY NODES
    for _, row in df.iterrows():
        person = str(row["Person"]).strip()
        gender = row.get("Gender", "Unknown")
        community = row.get("Community", "Unknown")
        G.add_node(person, gender=gender, community=community)

    # STEP 2: ADD EDGES + SECONDARY NODES
    for _, row in df.iterrows():
        person = str(row["Person"]).strip()
        interactions = str(row.get("Interacting Persons", "")).split(",")

        for item in interactions:
            item = item.strip()
            if not item:
                continue

            if "(" in item:
                name = item.split("(")[0].strip()
                rel = item.split("(")[1].replace(")", "").strip()
            else:
                name = item
                rel = "Other"

            if not name:
                continue

            if name not in G:
                G.add_node(name)

            G.add_edge(person, name, relationship=rel)

    # STEP 3: ATTRIBUTE INHERITANCE FIX
    for node in G.nodes():
        gender = G.nodes[node].get("gender")
        community = G.nodes[node].get("community")

        if gender not in ["M", "F"]:
            neighbors = list(G.neighbors(node))
            neighbor_genders = [
                G.nodes[n].get("gender")
                for n in neighbors
                if G.nodes[n].get("gender") in ["M", "F"]
            ]
            if neighbor_genders:
                G.nodes[node]["gender"] = Counter(neighbor_genders).most_common(1)[0][0]
            else:
                G.nodes[node]["gender"] = "M"

        if community is None or community == "Unknown":
            neighbors = list(G.neighbors(node))
            neighbor_comms = [
                G.nodes[n].get("community")
                for n in neighbors
                if G.nodes[n].get("community")
            ]
            if neighbor_comms:
                G.nodes[node]["community"] = Counter(neighbor_comms).most_common(1)[0][0]
            else:
                G.nodes[node]["community"] = "A"

    G_real = G
    
    fig = draw_graph(G_real, "community")
    metrics = get_metrics_text(G_real)
    implications = generate_implications(G_real)
    
    return fig_to_img(fig), metrics, implications

# ---------------- DRAW GRAPH ----------------
def draw_graph(G, color_mode="community"):
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    pos = nx.spring_layout(G, seed=42, k=2)

    # EDGE COLORS
    edge_colors = []
    for u, v in G.edges():
        rel = str(G[u][v].get("relationship", "")).lower()
        if "family" in rel:
            edge_colors.append("red")
        elif "neighbor" in rel or "neighbour" in rel:
            edge_colors.append("green")
        elif "friend" in rel:
            edge_colors.append("blue")
        elif "coworker" in rel or "co-worker" in rel:
            edge_colors.append("orange")
        elif "community" in rel:
            edge_colors.append("purple")
        else:
            edge_colors.append("gray")

    # NODE COLOR MODE
    if color_mode == "gender":
        attr = nx.get_node_attributes(G, "gender")
    elif color_mode == "community":
        attr = nx.get_node_attributes(G, "community")
    elif color_mode == "cluster":
        attr = nx.get_node_attributes(G, "community")
    else:
        attr = nx.get_node_attributes(G, "community")

    for n in G.nodes():
        if n not in attr:
            attr[n] = "Unknown"

    unique = list(set(attr.values()))
    mapping = {v: i for i, v in enumerate(unique)}
    colors = [mapping.get(attr[n], 0) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.Set3, node_size=500, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=ax)

    ax.axis("off")
    ax.set_title("Network Visualization", fontsize=14, fontweight='bold')
    
    return fig

# ---------------- DRAW EVENT GRAPH ----------------
def draw_event_graph(G, color_mode="community"):
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    persons = [n for n, d in G.nodes(data=True) if d.get("type") == "person"]
    events = [n for n, d in G.nodes(data=True) if d.get("type") == "event"]

    pos = {}
    for i, p in enumerate(persons):
        pos[p] = (0, i)
    for i, e in enumerate(events):
        pos[e] = (5, i)

    if color_mode == "gender":
        attr = nx.get_node_attributes(G, "gender")
    elif color_mode == "community":
        attr = nx.get_node_attributes(G, "community")
    else:
        attr = {n: "default" for n in persons}

    unique_vals = list(set(attr.values()))
    mapping = {v: i for i, v in enumerate(unique_vals)}
    person_colors = [mapping.get(attr.get(n, "default"), 0) for n in persons]

    nx.draw_networkx_nodes(G, pos, nodelist=persons, node_color=person_colors,
                           cmap=plt.cm.Set2, node_size=400, ax=ax, label="People")
    nx.draw_networkx_nodes(G, pos, nodelist=events, node_color="orange",
                           node_size=1200, ax=ax, label="Events")
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=ax)

    ax.legend()
    ax.axis("off")
    ax.set_title("Event Network Visualization", fontsize=14, fontweight='bold')
    
    return fig

# ---------------- DRAW PLACES GRAPH ----------------
def draw_places_graph(G, color_mode="community"):
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    persons = [n for n, d in G.nodes(data=True) if d.get("type") == "person"]
    places = [n for n, d in G.nodes(data=True) if d.get("type") == "place"]

    pos = {}
    for i, p in enumerate(persons):
        pos[p] = (0, i)
    for i, pl in enumerate(places):
        pos[pl] = (5, i)

    if color_mode == "gender":
        attr = nx.get_node_attributes(G, "gender")
    elif color_mode == "community":
        attr = nx.get_node_attributes(G, "community")
    else:
        attr = {n: "default" for n in persons}

    unique_vals = list(set(attr.values()))
    mapping = {v: i for i, v in enumerate(unique_vals)}
    person_colors = [mapping.get(attr.get(n, "default"), 0) for n in persons]

    nx.draw_networkx_nodes(G, pos, nodelist=persons, node_color=person_colors,
                           cmap=plt.cm.Set2, node_size=400, ax=ax, label="People")
    nx.draw_networkx_nodes(G, pos, nodelist=places, node_color="green",
                           node_size=1200, ax=ax, label="Places")
    nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black", ax=ax)

    ax.legend()
    ax.axis("off")
    ax.set_title("Places Network Visualization", fontsize=14, fontweight='bold')
    
    return fig

# ---------------- METRICS ----------------
def get_metrics_text(G):
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    density = nx.density(G)
    clustering = nx.average_clustering(G)

    is_connected = nx.is_connected(G)
    components = nx.number_connected_components(G)

    if is_connected:
        path_length = f"{nx.average_shortest_path_length(G):.3f}"
        diameter = nx.diameter(G)
    else:
        path_length = "Disconnected"
        diameter = "N/A"

    bet_cent = nx.betweenness_centrality(G)
    weighted_deg = dict(G.degree())

    hub = max(weighted_deg, key=weighted_deg.get)
    bridge = max(bet_cent, key=bet_cent.get)

    comm_attr = nx.get_node_attributes(G, "community")
    for n in G.nodes():
        if n not in comm_attr:
            comm_attr[n] = "Leader"

    partition_dict = defaultdict(list)
    for node, comm in comm_attr.items():
        partition_dict[comm].append(node)

    partition = [set(nodes_list) for nodes_list in partition_dict.values()]
    modularity_score = modularity(G, partition)
    num_communities = len(partition)

    try:
        assortativity = f"{nx.degree_assortativity_coefficient(G):.3f}"
    except:
        assortativity = "N/A"

    transitivity = nx.transitivity(G)

    return f"""
╔══════════════════════════════════════╗
║         NETWORK METRICS              ║
╠══════════════════════════════════════╣
║ BASIC                                ║
║   Nodes: {nodes}
║   Edges: {edges}
║   Density: {density:.3f}
╠══════════════════════════════════════╣
║ CONNECTIVITY                         ║
║   Connected: {is_connected}
║   Components: {components}
╠══════════════════════════════════════╣
║ PATHS                                ║
║   Avg Path Length: {path_length}
║   Diameter: {diameter}
╠══════════════════════════════════════╣
║ CLUSTERING                           ║
║   Local Clustering: {clustering:.3f}
║   Global (Transitivity): {transitivity:.3f}
╠══════════════════════════════════════╣
║ COMMUNITIES                          ║
║   Number of Communities: {num_communities}
║   Modularity: {modularity_score:.3f}
╠══════════════════════════════════════╣
║ CENTRALITY                           ║
║   Most Influential (Hub): {hub}
║   Bridge Node: {bridge}
╠══════════════════════════════════════╣
║ STRUCTURE                            ║
║   Assortativity: {assortativity}
╚══════════════════════════════════════╝
"""

# ---------------- IMPLICATIONS ----------------
def generate_implications(G):
    desc = []

    density = nx.density(G)
    clustering = nx.average_clustering(G)
    components = nx.number_connected_components(G)

    bet_cent = nx.betweenness_centrality(G)
    weighted_deg = dict(G.degree())

    hub = max(weighted_deg, key=weighted_deg.get)
    bridge = max(bet_cent, key=bet_cent.get)

    comm_attr = nx.get_node_attributes(G, "community")
    for n in G.nodes():
        if n not in comm_attr:
            comm_attr[n] = "Leader"

    num_communities = len(set(comm_attr.values()))

    if clustering > 0.6:
        desc.append("🔗 **Strong clan-based bonding** - The network shows tight-knit groups with strong internal connections.")
    elif clustering > 0.3:
        desc.append("🤝 **Moderate cohesion** - Social groups exist but aren't extremely tight-knit.")
    else:
        desc.append("📡 **Weak bonding** - Interactions are sparse with limited clustering.")

    if components > 1:
        desc.append(f"⚠️ **Fragmented Network** - The network has {components} disconnected components.")
    else:
        desc.append("✅ **Fully Connected** - All members can reach each other through the network.")

    if weighted_deg[hub] > (sum(weighted_deg.values()) / len(weighted_deg)) * 1.5:
        desc.append(f"👑 **Dominant Leader** - {hub} acts as a central authority figure.")
    else:
        desc.append("🏛️ **Distributed Leadership** - No single dominant leader; power is shared.")

    if bet_cent[bridge] > 0.2:
        desc.append(f"🌉 **Strong Bridge Node** - {bridge} is critical for connecting different groups.")
    else:
        desc.append(f"🔀 **Supporting Connector** - {bridge} helps facilitate communication between groups.")

    if num_communities >= 4:
        desc.append("🏘️ **Multiple Communities** - Several distinct social groups exist within the network.")
    elif num_communities > 1:
        desc.append("👥 **Some Community Structure** - A few identifiable groups are present.")
    else:
        desc.append("🔄 **Homogeneous Network** - No strong community divisions detected.")

    if nx.is_connected(G):
        path_length = nx.average_shortest_path_length(G)
        if path_length < 2.5:
            desc.append("⚡ **Fast Information Spread** - News and information travel quickly through the network.")
        elif path_length < 4:
            desc.append("📮 **Moderate Information Flow** - Information spreads at a reasonable pace.")
        else:
            desc.append("🐌 **Slow Communication** - Information takes time to reach all members.")

    centralization = max(nx.degree_centrality(G).values())
    if centralization > 0.4:
        desc.append("🎯 **Highly Centralized** - The network depends heavily on key individuals.")
    else:
        desc.append("🛡️ **Decentralized & Resilient** - The network can withstand the loss of individual members.")

    return "\n\n".join(desc)

# ---------------- SYNTHETIC NETWORK ----------------
def generate_synthetic(node_count, color_mode):
    global G_real, G_syn

    if G_real is None:
        return None, "Please load survey data first.", ""

    try:
        N = int(node_count)
    except:
        N = 30

    avg_deg = sum(dict(G_real.degree()).values()) / G_real.number_of_nodes()
    k = max(2, int(avg_deg))
    if k % 2 != 0:
        k += 1

    G_syn = nx.watts_strogatz_graph(N, k, 0.2)

    real_genders = list(nx.get_node_attributes(G_real, "gender").values())
    real_communities = list(nx.get_node_attributes(G_real, "community").values())

    for i, node in enumerate(G_syn.nodes()):
        gender = real_genders[i % len(real_genders)] if real_genders else "Unknown"
        community = real_communities[i % len(real_communities)] if real_communities else "Unknown"
        G_syn.nodes[node]["gender"] = gender
        G_syn.nodes[node]["community"] = community

    fig = draw_graph(G_syn, color_mode)
    metrics = get_metrics_text(G_syn)
    implications = generate_implications(G_syn)
    
    return fig_to_img(fig), metrics, implications

# ---------------- LOAD EVENT NETWORK ----------------
def load_event_network(file, color_mode):
    global G_event

    if file is None:
        return None, "Please upload an Excel file.", ""

    df = pd.read_excel(file.name)
    G = nx.Graph()

    for _, row in df.iterrows():
        person = str(row["Person"]).strip()
        gender = row.get("Gender", "Unknown")
        community = row.get("Community", "Unknown")

        G.add_node(person, type="person", gender=gender, community=community)
        events = str(row["Events"]).split(",")

        for e in events:
            event = e.strip()
            if event:
                G.add_node(event, type="event")
                G.add_edge(person, event)

    G_event = G
    
    fig = draw_event_graph(G, color_mode)
    metrics = get_metrics_text(G)
    implications = generate_implications(G)
    
    return fig_to_img(fig), metrics, implications

# ---------------- LOAD PLACES NETWORK ----------------
def load_places_network(file, color_mode):
    global G_places

    if file is None:
        return None, "Please upload an Excel file.", ""

    df = pd.read_excel(file.name)
    G = nx.Graph()

    for _, row in df.iterrows():
        person = str(row["Person"]).strip()
        gender = row.get("Gender", "Unknown")
        community = row.get("Community", "Unknown")

        G.add_node(person, type="person", gender=gender, community=community)
        places = str(row["Places Visited"]).split(",")

        for p in places:
            place = p.strip()
            if place:
                G.add_node(place, type="place")
                G.add_edge(person, place)

    G_places = G
    
    fig = draw_places_graph(G, color_mode)
    metrics = get_metrics_text(G)
    implications = generate_implications(G)
    
    return fig_to_img(fig), metrics, implications

# ---------------- UPDATE COLOR MODE ----------------
def update_real_network_color(color_mode):
    global G_real
    if G_real is None:
        return None
    fig = draw_graph(G_real, color_mode)
    return fig_to_img(fig)

def update_synthetic_network_color(color_mode):
    global G_syn
    if G_syn is None:
        return None
    fig = draw_graph(G_syn, color_mode)
    return fig_to_img(fig)

def update_event_network_color(color_mode):
    global G_event
    if G_event is None:
        return None
    fig = draw_event_graph(G_event, color_mode)
    return fig_to_img(fig)

def update_places_network_color(color_mode):
    global G_places
    if G_places is None:
        return None
    fig = draw_places_graph(G_places, color_mode)
    return fig_to_img(fig)

# ---------------- GRADIO INTERFACE ----------------
def create_interface():
    
    with gr.Blocks(title="Tribal Network Analyzer", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🌐 Tribal Network Analyzer
        **Analyze and visualize social networks from survey data**
        
        ---
        """)
        
        with gr.Row():
            # LEFT PANEL - Controls
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Controls")
                
                color_mode = gr.Radio(
                    choices=["community", "gender", "cluster"],
                    value="community",
                    label="🎨 Color Mode"
                )
                
                gr.Markdown("---")
                gr.Markdown("### 📊 Network Metrics")
                metrics_output = gr.Textbox(
                    label="",
                    lines=20,
                    interactive=False
                )
                
                gr.Markdown("---")
                gr.Markdown("### 💡 Social Implications")
                implications_output = gr.Markdown("")
                
                gr.Markdown("---")
                gr.Markdown("""
                ### 🎨 Edge Color Legend
                - 🔴 **Red** → Family
                - 🟢 **Green** → Neighbor
                - 🔵 **Blue** → Friend
                - 🟠 **Orange** → Coworker
                - 🟣 **Purple** → Community Member
                - ⚫ **Gray** → Other
                """)
            
            # RIGHT PANEL - Graphs
            with gr.Column(scale=2):
                with gr.Tabs():
                    # TAB 1: Real Network
                    with gr.Tab("📈 Real Network"):
                        gr.Markdown("#### Upload Survey Excel File")
                        survey_file = gr.File(
                            label="Survey Data (.xlsx)",
                            file_types=[".xlsx"]
                        )
                        load_survey_btn = gr.Button("🔄 Load Survey Data", variant="primary")
                        real_network_img = gr.Image(label="Network Visualization", type="filepath")
                    
                    # TAB 2: Synthetic Network
                    with gr.Tab("🔮 Synthetic Network"):
                        gr.Markdown("#### Generate Synthetic Network")
                        gr.Markdown("*Load survey data first to generate a synthetic network with similar properties*")
                        node_count = gr.Number(value=30, label="Number of Nodes", precision=0)
                        generate_syn_btn = gr.Button("⚡ Generate Synthetic Network", variant="secondary")
                        synthetic_network_img = gr.Image(label="Synthetic Network Visualization", type="filepath")
                    
                    # TAB 3: Event Network
                    with gr.Tab("📅 Event Network"):
                        gr.Markdown("#### Upload Event Data Excel File")
                        gr.Markdown("*Expected columns: Person, Gender, Community, Events*")
                        event_file = gr.File(
                            label="Event Data (.xlsx)",
                            file_types=[".xlsx"]
                        )
                        load_event_btn = gr.Button("🔄 Load Event Data", variant="primary")
                        event_network_img = gr.Image(label="Event Network Visualization", type="filepath")
                    
                    # TAB 4: Places Network
                    with gr.Tab("📍 Places Network"):
                        gr.Markdown("#### Upload Places Data Excel File")
                        gr.Markdown("*Expected columns: Person, Gender, Community, Places Visited*")
                        places_file = gr.File(
                            label="Places Data (.xlsx)",
                            file_types=[".xlsx"]
                        )
                        load_places_btn = gr.Button("🔄 Load Places Data", variant="primary")
                        places_network_img = gr.Image(label="Places Network Visualization", type="filepath")
        
        # EVENT HANDLERS
        load_survey_btn.click(
            fn=load_survey,
            inputs=[survey_file],
            outputs=[real_network_img, metrics_output, implications_output]
        )
        
        generate_syn_btn.click(
            fn=generate_synthetic,
            inputs=[node_count, color_mode],
            outputs=[synthetic_network_img, metrics_output, implications_output]
        )
        
        load_event_btn.click(
            fn=load_event_network,
            inputs=[event_file, color_mode],
            outputs=[event_network_img, metrics_output, implications_output]
        )
        
        load_places_btn.click(
            fn=load_places_network,
            inputs=[places_file, color_mode],
            outputs=[places_network_img, metrics_output, implications_output]
        )
        
        # Color mode changes
        color_mode.change(
            fn=update_real_network_color,
            inputs=[color_mode],
            outputs=[real_network_img]
        )
        
        color_mode.change(
            fn=update_synthetic_network_color,
            inputs=[color_mode],
            outputs=[synthetic_network_img]
        )
        
        color_mode.change(
            fn=update_event_network_color,
            inputs=[color_mode],
            outputs=[event_network_img]
        )
        
        color_mode.change(
            fn=update_places_network_color,
            inputs=[color_mode],
            outputs=[places_network_img]
        )
        
        gr.Markdown("""
        ---
        ### 📋 Expected Excel Format
        
        | Column | Description |
        |--------|-------------|
        | **Person** | Name of the person |
        | **Gender** | M or F |
        | **Community** | Community identifier |
        | **Interacting Persons** | Comma-separated list: `Name1(Family), Name2(Friend)` |
        | **Events** | (For event data) Comma-separated list of events |
        | **Places Visited** | (For places data) Comma-separated list of places |
        """)
    
    return app

# ---------------- LAUNCH ----------------
if __name__ == "__main__":
    app = create_interface()
    app.launch(share=True)  # share=True generates a public link
