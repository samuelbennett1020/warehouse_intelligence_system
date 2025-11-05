from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib.lines import Line2D


# --------------------------
# 1. Data Loading & Cleaning
# --------------------------
def load_and_clean_data(parquest_path: str) -> pd.DataFrame:
    """
    Load barcode data and clean it by combining actual and expected locations.

    Returns:
        pd.DataFrame: Cleaned dataframe with 'reported_location' and 'misplaced' columns.
    """
    df = pd.read_parquet(parquest_path)
    df = df.dropna(subset=["barcode_actually_scanned_at", "barcode_should_be_at"], how="all")
    df["reported_location"] = df["barcode_actually_scanned_at"].combine_first(df["barcode_should_be_at"])
    df["misplaced"] = df["reported_location"] != df["location"]
    return df


# --------------------------
# 2. Pair Counting Helper
# --------------------------
def count_pairs(grouped_series: pd.Series) -> dict[tuple[str, str], int]:
    """
    Count occurrences of unique pairs of items in grouped series.

    Args:
        grouped_series (pd.Series): Series of lists of items to pair.

    Returns:
        dict[tuple[str, str], int]: Dictionary with pair tuples as keys and counts as values.
    """
    pair_counter: Counter[tuple[str, str]] = Counter()

    for items in grouped_series:
        # Filter out NaN once and convert to string
        clean_items = [str(item) for item in items if pd.notna(item)]
        n = len(clean_items)
        if n < 2:
            continue  # skip groups with less than 2 items
        # Generate pairs directly
        for i in range(n):
            for j in range(i + 1, n):
                a, b = clean_items[i], clean_items[j]
                # Ensure consistent ordering (a < b)
                if a > b:
                    a, b = b, a
                pair_counter[(a, b)] += 1

    return dict(pair_counter)


# --------------------------
# 3. Generate Pair Counts
# --------------------------
def generate_pair_counts(df: pd.DataFrame, picked_threshold: int = 2) -> tuple[dict, dict, dict]:
    """
    Generate co-located, misplaced, and picked-together pair counts.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
        picked_threshold (int): Minimum count for a pair to be considered "picked together".

    Returns:
        tuple[dict, dict, dict]: co-location counts, misplacement counts, picked-together counts
    """
    # Co-located pairs
    grouped_actual = df.groupby(["timestep", "reported_location"])["barcode"].apply(list)
    co_location_counts = count_pairs(grouped_actual)

    # misplaced pairs
    grouped_misplaced = df[df["misplaced"]].groupby(["timestep", "reported_location"])["barcode"].apply(list)
    misplacement_counts = count_pairs(grouped_misplaced)

    # Picked-together pairs (co-movement over time)
    grouped_time = df.groupby("timestep")["barcode"].apply(list)
    picked_counts = count_pairs(grouped_time)
    picked_pairs = {pair: count for pair, count in picked_counts.items() if count >= picked_threshold}

    return co_location_counts, misplacement_counts, picked_pairs


# --------------------------
# 4. Display Top Pairs & Items
# --------------------------
def print_top_pairs(count_dict: dict, title: str, top_n: int = 5) -> None:
    """
    Print the top N pairs from a count dictionary.

    Args:
        count_dict (dict): Dictionary of pair counts.
        title (str): Description title for the pairs.
        top_n (int): Number of top pairs to display.
    """
    sorted_pairs = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    print(f"Top {top_n} {title} pairs:")
    for (a, b), count in sorted_pairs:
        print(f"{a}-{b}: {count}")


def print_top_items(df: pd.DataFrame, top_n: int = 5) -> None:
    """
    Print top misplaced items and most active/picked items.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
        top_n (int): Number of top items to display.
    """
    # Most misplaced
    misplaced_counts = df[df["misplaced"]]["barcode"].value_counts()
    print(f" Top {top_n} most misplaced items:")
    print(misplaced_counts.head(top_n))

    # Most active/picked items
    picked_counts = df.groupby("barcode")["timestep"].count()
    print(f" Top {top_n} most picked / active items:")
    print(picked_counts.sort_values(ascending=False).head(top_n))


# --------------------------
# 5. Build misplaced Graph
# --------------------------
def build_misplaced_graph(df: pd.DataFrame, misplacement_counts: dict) -> nx.Graph:
    """
    Build a NetworkX graph of misplaced items and their pair relationships.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
        misplacement_counts (dict): Counts of misplaced pairs.

    Returns:
        nx.Graph: misplaced items graph.
    """
    G = nx.Graph()
    # Add all misplaced items as nodes
    all_misplaced_items = df[df["misplaced"]]["barcode"].unique()
    G.add_nodes_from(all_misplaced_items)
    # Add edges for misplaced pairs
    for pair, count in misplacement_counts.items():
        G.add_edge(pair[0], pair[1], weight=count)
    return G


# --------------------------
# 6. Visualize misplaced Graph
# --------------------------
def visualize_misplaced_graph(G: nx.Graph, df: pd.DataFrame, top_n: int = 5) -> None:
    """
    Visualize a graph of misplaced items with top items highlighted.

    Args:
        G (nx.Graph): misplaced items graph.
        df (pd.DataFrame): Cleaned dataframe.
        top_n (int): Number of top misplaced items to highlight.
    """
    plt.figure(figsize=(10, 8))

    # Node positions
    pos = nx.spring_layout(G, seed=42, weight="weight")

    # Highlight top misplaced items
    misplaced_items_count = df[df["misplaced"]]["barcode"].value_counts()
    top_nodes = [node for node in misplaced_items_count.head(top_n).index if node in G.nodes()]
    node_colors = ["orange" if node in top_nodes else "lightblue" for node in G.nodes()]

    # Edge styling
    edge_colors = ["red" for _ in G.edges()]
    edge_widths = [(d["weight"] ** 0.6) * 2 for _, _, d in G.edges(data=True)]

    # Draw graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        edge_color=edge_colors,
        width=edge_widths,
        node_size=1000,
        font_size=10,
        font_weight="bold"
    )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Top misplaced Item',
               markerfacecolor='orange', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Other misplaced Items',
               markerfacecolor='lightblue', markersize=10),
        Line2D([0], [0], color="red", lw=2, label="misplaced Pair")
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.title("Warehouse misplaced Items and Pairs")
    plt.show()


# --------------------------
# 7. Main Execution
# --------------------------
def main(parquet_path: str = "../../data/processed_data/timeseries_data.parquet") -> None:
    df = load_and_clean_data(parquet_path)
    co_location_counts, misplacement_counts, picked_pairs = generate_pair_counts(df)

    # Print results
    print_top_pairs(co_location_counts, "co-located")
    print_top_pairs(misplacement_counts, "misplaced together")
    print_top_pairs(picked_pairs, "picked together")
    print_top_items(df)

    # Build and visualize graph
    G_misplaced = build_misplaced_graph(df, misplacement_counts)
    visualize_misplaced_graph(G_misplaced, df)


if __name__ == "__main__":
    main()
