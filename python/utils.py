import json
from pathlib import Path
import numpy as np
import torch
from typing import Optional, Union, Tuple, List, Dict, Any
from torch_geometric.data import HeteroData
from torch_geometric.explain import Explainer, HeteroExplanation
from torch_geometric.explain.config import ExplanationType, ModelMode
from numpy.lib import recfunctions as rfn


class CadenceEncoder(object):
    """
    Encodes cadences to integer labels.

    The accepted cadences are:
    - PAC (Perfect Authentic Cadence)
    - IAC (Imperfect Authentic Cadence)
    - HC (Half Cadence)
    - DC (Deceptive Cadence)
    - EC (Evaded Cadence)
    - PC (Plagal Cadence)

    The encoding is:
    - No cadence: 0
    - PAC: 1
    - IAC: 2
    - HC: 3
    - DC/EC/PC: 4 (all grouped together because they are sparse in datasets)
    """
    def __init__(self):
        self.cadences = {
            "": 0,
            "PAC": 1,
            "IAC": 2,
            "HC": 3,
            "DC": 4,
            "EC": 4,
            "PC": 4,
        }
        self.accepted_cadences = np.array(["", "PAC", "IAC", "HC", "DC/EC/PC"])
        self.encode_dim = len(np.unique(list(self.cadences.values())))

    def encode(self, note_array, cadences):
        """
        Encodes a note array with cadences to integer labels.

        Parameters
        ----------
        note_array : numpy structured array
            A note array.
        cadences : list
            A list of partitura.Cadence objects.

        """
        labels = torch.zeros(len(note_array), dtype=torch.long)
        for cadence in cadences:
            labels[note_array["onset_div"] == cadence.start.t] = self.cadences[cadence.text]
        return labels

    def decode(self, x):
        """
        Decodes integer labels to cadences.

        Parameters
        ----------
        x: numpy array or torch tensor
            Cadence Integer labels.

        Returns
        -------
        out: numpy array
            Cadence strings.
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return self.accepted_cadences[x]


def hetero_graph_from_note_array(note_array, rest_array=None, norm2bar=False, pot_edge_dist=3):
    '''Turn note_array to homogeneous graph dictionary.

    Parameters
    ----------
    note_array : structured array
        The partitura note_array object. Every entry has 5 attributes, i.e. onset_time, note duration, note velocity, voice, id.
    rest_array : structured array
        A structured rest array similar to the note array but for rests.
    t_sig : list
        A list of time signature in the piece.
    '''

    edg_src = list()
    edg_dst = list()
    etype = list()
    pot_edges = list()
    start_rest_index = len(note_array)
    for i, x in enumerate(note_array):
        for j in np.where(note_array["onset_div"] == x["onset_div"])[0]:
            if i != j:
                edg_src.append(i)
                edg_dst.append(j)
                etype.append(0)
        if pot_edge_dist:
            for j in np.where(
                    (note_array["onset_div"] > x["onset_div"]+x["duration_div"]) &
                    (note_array["onset_beat"] <= x["onset_beat"] + x["duration_beat"] + pot_edge_dist*x["ts_beats"])
            )[0]:
                pot_edges.append([i, j])
        for j in np.where(note_array["onset_div"] == x["onset_div"] + x["duration_div"])[0]:
            edg_src.append(i)
            edg_dst.append(j)
            etype.append(1)

        if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
            for j in np.where(rest_array["onset_div"] == x["onset_div"] + x["duration_div"])[0]:
                edg_src.append(i)
                edg_dst.append(j + start_rest_index)
                etype.append(1)

        for j in np.where(
                (x["onset_div"] < note_array["onset_div"]) & (x["onset_div"] + x["duration_div"] > note_array["onset_div"]))[0]:
            edg_src.append(i)
            edg_dst.append(j)
            etype.append(2)

    if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
        for i, r in enumerate(rest_array):
            for j in np.where(np.isclose(note_array["onset_div"], r["onset_div"] + r["duration_div"], rtol=1e-04, atol=1e-04) == True)[0]:
                edg_src.append(start_rest_index + i)
                edg_dst.append(j)
                etype.append(1)

        feature_fn = [dname for dname in note_array.dtype.names if dname not in rest_array.dtype.names]
        if feature_fn:
            rest_feature_zeros = np.zeros((len(rest_array), len(feature_fn)))
            rest_feature_zeros = rfn.unstructured_to_structured(rest_feature_zeros, dtype=list(map(lambda x: (x, '<4f'), feature_fn)))
            rest_array = rfn.merge_arrays((rest_array, rest_feature_zeros))
    else:
        end_times = note_array["onset_div"] + note_array["duration_div"]
        for et in np.sort(np.unique(end_times))[:-1]:
            if et not in note_array["onset_div"]:
                scr = np.where(end_times == et)[0]
                diffs = note_array["onset_div"] - et
                tmp = np.where(diffs > 0, diffs, np.inf)
                dst = np.where(tmp == tmp.min())[0]
                for i in scr:
                    for j in dst:
                        edg_src.append(i)
                        edg_dst.append(j)
                        etype.append(3)

    edges = np.array([edg_src, edg_dst, etype])

    # Resize Onset Beat to bar
    if norm2bar:
        note_array["onset_beat"] = np.mod(note_array["onset_beat"], note_array["ts_beats"])
        if isinstance(rest_array, np.ndarray) and rest_array.size > 0:
            rest_array["onset_beat"] = np.mod(rest_array["onset_beat"], rest_array["ts_beats"])

    nodes = np.hstack((note_array, rest_array))
    if pot_edge_dist:
        pot_edges = np.hstack((np.array(pot_edges).T, edges[:, edges[2] == 1][:2]))
        return nodes, edges, pot_edges
    return nodes, edges


def hetero_fidelity(
    explainer: Explainer,
    explanation: HeteroExplanation,
) -> Tuple[float, float]:
    r"""Evaluates the fidelity of an
    :class:`~torch_geometric.explain.Explainer` given an
    :class:`~torch_geometric.explain.HeteroExplanation`, as described in the
    `"GraphFramEx: Towards Systematic Evaluation of Explainability Methods for
    Graph Neural Networks" <https://arxiv.org/abs/2206.09677>`_ paper.

    Fidelity evaluates the contribution of the produced explanatory subgraph
    to the initial prediction, either by giving only the subgraph to the model
    (fidelity-) or by removing it from the entire graph (fidelity+).
    The fidelity scores capture how good an explainable model reproduces the
    natural phenomenon or the GNN model logic.

    For **phenomenon** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= \frac{1}{N} \sum_{i = 1}^N
        \| \mathbb{1}(\hat{y}_i = y_i) -
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = y_i) \|

        \textrm{fid}_{-} &= \frac{1}{N} \sum_{i = 1}^N
        \| \mathbb{1}(\hat{y}_i = y_i) -
        \mathbb{1}( \hat{y}_i^{G_S} = y_i) \|

    For **model** explanations, the fidelity scores are given by:

    .. math::
        \textrm{fid}_{+} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_{C \setminus S}} = \hat{y}_i)

        \textrm{fid}_{-} &= 1 - \frac{1}{N} \sum_{i = 1}^N
        \mathbb{1}( \hat{y}_i^{G_S} = \hat{y}_i)

    Args:
        explainer (Explainer): The explainer to evaluate.
        explanation (HeteroExplanation): The explanation to evaluate.

    This implementation was adapted from the original implementation in the torch_geometric library to work with the HeteroExplanation class.
    """
    if explainer.model_config.mode == ModelMode.regression:
        raise ValueError("Fidelity not defined for 'regression' models")

    edge_mask = {k: (explanation[k].edge_mask > 0).long() for k in explanation.edge_types}
    node_mask = {k: (explanation[k].node_mask > 0).long() for k in explanation.node_types}
    kwargs = {}
    # kwargs = {key: explanation[key] for key in explanation._model_args}

    y = explanation.target
    if explainer.explanation_type == ExplanationType.phenomenon:
        y_hat = explainer.get_prediction(
            explanation.x_dict,
            explanation.edge_index_dict,
            **kwargs,
        )
        y_hat = explainer.get_target(y_hat)

    explain_y_hat = explainer.get_masked_prediction(
        explanation.x_dict,
        explanation.edge_index_dict,
        node_mask,
        edge_mask,
        # **kwargs,
    )
    explain_y_hat = explainer.get_target(explain_y_hat)

    complement_y_hat = explainer.get_masked_prediction(
        explanation.x_dict,
        explanation.edge_index_dict,
        {k: 1 - m for k, m in node_mask.items()},
        {k: 1 - m for k, m in edge_mask.items()},
        # **kwargs,
    )
    complement_y_hat = explainer.get_target(complement_y_hat)

    if explanation.get('index') is not None:
        y = y[explanation.index]
        if explainer.explanation_type == ExplanationType.phenomenon:
            y_hat = y_hat[explanation.index]
        explain_y_hat = explain_y_hat[explanation.index]
        complement_y_hat = complement_y_hat[explanation.index]

    if explainer.explanation_type == ExplanationType.model:
        pos_fidelity = 1. - (complement_y_hat == y).float().mean()
        neg_fidelity = 1. - (explain_y_hat == y).float().mean()
    else:
        pos_fidelity = ((y_hat == y).float() -
                        (complement_y_hat == y).float()).abs().mean()
        neg_fidelity = ((y_hat == y).float() -
                        (explain_y_hat == y).float()).abs().mean()

    return float(pos_fidelity), float(neg_fidelity)


def add_reverse_edges(graph, mode):
    """
    Add reverse edges to the graph.

    Parameters
    ----------
    graph : HeteroData
        The graph object.
    mode : str
        The mode of adding reverse edges. Either 'new_type' or 'undirected'.
    """
    if mode == "new_type":
        # add reversed consecutive edges
        graph["note", "consecutive_rev", "note"].edge_index = graph[
            "note", "consecutive", "note"
        ].edge_index[[1, 0]]
        # add reversed during edges
        graph["note", "during_rev", "note"].edge_index = graph[
            "note", "during", "note"
        ].edge_index[[1, 0]]
        # add reversed rest edges
        graph["note", "rest_rev", "note"].edge_index = graph[
            "note", "rest", "note"
        ].edge_index[[1, 0]]
    else:
        raise ValueError("mode must be either 'new_type' or 'undirected'")
    return graph


def create_score_graph(
        features: Union[np.ndarray, torch.Tensor],
        note_array: np.ndarray,
        labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
        add_reverse: bool= True,
    ):

    nodes, edges = hetero_graph_from_note_array(note_array)
    edges = edges[:2]
    edge_types = edges[-1]


    edge_etypes = {
        0: "onset",
        1: "consecutive",
        2: "during",
        3: "rest"
    }
    edges = torch.from_numpy(edges).long()
    edge_types = torch.from_numpy(edge_types).long()
    graph = HeteroData()
    graph["note"].x = torch.from_numpy(features).float() if isinstance(features, np.ndarray) else features.float()
    if labels is not None:
        graph["note"].y = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else labels.long()
    graph["note"].onset_div = torch.from_numpy(note_array['onset_div']).long()
    graph["note"].duration_div = torch.from_numpy(note_array['duration_div']).long()
    graph["note"].pitch = torch.from_numpy(note_array['pitch']).long()
    for k, v in edge_etypes.items():
        graph['note', v, 'note'].edge_index = edges[:, edge_types == k]

    if add_reverse:
        graph = add_reverse_edges(graph, mode="new_type")

    return graph

def save_pyg_graph_as_json(graph, ids, extra_info=None, path="./"):
    """Save the graph as a json file.

    Args:
        graph (torch_geometric.data.HeteroData): the graph to save
    """
    out_dict = {}
    # for k,v in graph.__dict__.items():
    #     if isinstance(v, (np.ndarray,torch.Tensor)):
    #         out_dict[k] = v.tolist()
    #     elif isinstance(v, str):
    #         out_dict[k] = v
    # export the input edges
    for k, v in graph.edge_index_dict.items():
        out_dict[k[1]] = v.tolist()

    # export the output edges
    # truth edges
    # out_dict["output_edges_dict"]["truth"] = graph["truth_edges"].tolist()
    # # potential edges
    # out_dict["output_edges_dict"]["potential"] = graph["pot_edges"].tolist()

    # export the nodes ids
    if "_" in ids[0]:  # MEI with multiple parts, remove the Pxx_ prefix
        out_dict["id"] = [i.split("_")[1] for i in ids]
    else:
        out_dict["id"] = ids.tolist()

    if extra_info is not None:
        for k, v in extra_info.items():
            out_dict[k] = v

    with open(Path(path, graph.name + ".json"), "w") as f:
        print("Saving graph to", Path(path, graph.name + ".json"))
        json.dump(out_dict, f)