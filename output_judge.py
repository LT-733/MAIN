import numpy as np
import transformers
from sentence_transformers import SentenceTransformer, SimilarityFunction
import torch
import clustering
import matplotlib.pyplot as plt
import get_outputs
import math

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
print(f"using {device} as performance accelerator")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device, similarity_fn_name=SimilarityFunction.COSINE)
model = transformers.AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

model = model.to(device=device)
# tokenizer = tokenizer.to(device=device)

text_outputs: list[dict] = get_outputs.get_chat_content()
baseline_ans: str = "In computer science, a linked list is a linear collection of data elements whose order is not given by their physical placement in memory. Instead, each element points to the next. It is a data structure consisting of a collection of nodes which together represent a sequence. In its most basic form, each node contains data, and a reference (in other words, a link) to the next node in the sequence. This structure allows for efficient insertion or removal of elements from any position in the sequence during iteration."

# def get_stagnation(self_similarity: list[list], baseline_similarity: list[list]) -> list[int]:
#     """Flags a row as stagnated if its row variance is way too low or its mean is way too high"""
#     off_diagonal: list = []
#     for i in range(len(self_similarity)):
#         for j in range(len(self_similarity[0])):
#             if i != j:
#                 off_diagonal.append(self_similarity[i][j])
#     mean = sum(off_diagonal) / len(off_diagonal)
#     variance = sum((i-mean)**2 for i in off_diagonal) / len(off_diagonal)
#     std = math.sqrt(variance)
#     theta: float = std * 0.5 + mean
#     # k: int = 5 if len(baseline_similarity) >= 5 else len(baseline_similarity)
#     # variance_list: list = []
#     # for l in range(k):
#     #     rowmean = sum(baseline_similarity[l]) / len(baseline_similarity[l])
#     #     rowvariance = sum((m - rowmean)**2 for m in baseline_similarity[l]) / len(baseline_similarity[l])
#     #     variance_list.append(rowvariance)
#     # # this is a magic number that we need to change or fine tune
#     # safety_scale: float = 0.04
#     # epsilon: float = np.median(variance_list) * safety_scale
#     stagnated_rows:list[int] = []
#     for x in range(len(baseline_similarity)):
#         curmean = sum(baseline_similarity[x]) / len(baseline_similarity[x])
#         # curvariance = sum((cur - curmean)**2 for cur in baseline_similarity[x]) / len(baseline_similarity[x])
#         if curmean > theta:
#             stagnated_rows.append(x)
#     return stagnated_rows

def gramschmidt_process(embedding_matrix: list[list], device, noise_threshold: float=1e-5):
    orthogonal_basis: list = []
    results: list[float] = []
    for t in range(len(embedding_matrix)):
        if t == 0:
            curvector = torch.tensor(embedding_matrix[t]).to(device=device)
            results.append(torch.norm(curvector, dim=0).item())
            orthogonal_basis.append(torch.nn.functional.normalize(curvector, dim=0))
        else:
            curvector = torch.tensor(embedding_matrix[t]).to(device=device)
            bases_matrix = torch.stack(orthogonal_basis)
            projection_results = bases_matrix @ curvector
            parallel_vector = bases_matrix.T @ projection_results
            residual = curvector - parallel_vector
            norm_res = torch.norm(residual, dim=0).item()
            results.append(norm_res)
            if norm_res > noise_threshold:
                orthogonal_basis.append(torch.nn.functional.normalize(residual, dim=0))
            else:
                orthogonal_basis.append(torch.zeros(384, dtype=curvector.dtype, device=device))
    return plt.plot(results)



def get_semantic_drift(tst_model, given_tokenizer, outputs: list[dict], ans: str):
    activation_storage: dict = {}
    target_layer = tst_model.encoder.layer[5]
    model_names = [output['model'] for output in outputs]
    def callback_function(module, in_tensor, out_tensor, model_name: str):
        activation_storage[model_name] = out_tensor.detach().cpu()
    # This is a placeholder we will fill in the PyTorch loop
    curmodel: str = ""

    with torch.no_grad():
        # first we extract the last layer of the embedding model to get the tensor coming out of it, so we can compare the semantic drift from the answer.
        for candidate in outputs:
            curmodel = candidate["model"]
            hook = target_layer.register_forward_hook(lambda m, i, o, name=curmodel: callback_function(m, i, o, name))
            tokenized = given_tokenizer(candidate["text"], return_tensors="pt")
            tokenized = tokenized.to(device=device) # send the memory to the accelerator
            # that double asterisk distributes the key values pairs for you
            _ = tst_model(**tokenized)
            hook.remove()
            del _
        # must also get the embedding of the final layer for our baseline answer
        curmodel = "answer"
        anshook = target_layer.register_forward_hook(lambda m, i, o, name=curmodel: callback_function(m, i, o, name))
        ans_tokenized = given_tokenizer(ans, return_tensors="pt")
        ans_tokenized = ans_tokenized.to(device=device)
        _ = tst_model(**ans_tokenized)
        anshook.remove()
        del _
    
    activation_storage["answer"] = activation_storage["answer"].squeeze(0)
    activation_storage["answer"] = torch.nn.functional.normalize(activation_storage["answer"])
    final_drifts: list[tuple[str, list, list]] = []
    for name, data in (activation_storage).items():
        if name == "answer": 
            continue
        else:
            data = data.squeeze(0)
            tokenmap = data
            data = torch.nn.functional.normalize(data)
            # final = (data @ activation_storage["answer"].T).max(dim=0, keepdim=True)[0]
            final = (data @ activation_storage["answer"].T)
            # self_sim = data @ data.T
            # self_sim = self_sim.tolist()
            final = final.tolist()
            final_drifts.append((name, final, tokenmap))
    fig, axs = plt.subplots(ncols=len(final_drifts), layout="constrained", nrows=1, squeeze=False, figsize=(len(final_drifts)*3.0, 2.5))
    fig.suptitle(f"Semantic Drift From the Baseline Answer of {len(final_drifts)} Models")
    for i in range(len(final_drifts)):
        # stagnated: list = get_stagnation(final_drifts[i][2], final_drifts[i][1])
        axs[0, i].imshow(final_drifts[i][1], cmap="viridis", vmin=0.7, vmax=1.0, aspect="auto")
        # axs[0, i].hlines(stagnated, 0, len(final_drifts[i][1][0]), color='r')
        # print(stagnated)
        axs[0, i].set_title(final_drifts[i][0])
        axs[0, i].set_xlabel("Baseline")
        if i == 0:
            axs[0, i].set_ylabel("Response")
    return {
        "figure": fig,
        "drifts": final_drifts
    }





def get_distance_matrix(model_out: list[dict], std_ans: str) -> tuple[list[list], list]:

    embeddings: list[torch.Tensor] = []

    for output in model_out:
        embedding = encoder.encode(output["text"], convert_to_tensor=True)
        embeddings.append(embedding)
    embedded_ans: torch.Tensor = encoder.encode(std_ans, convert_to_tensor=True)
    distances: list[list] = []
    distances_to_ans = []
    for i in range(len(embeddings)):
        toans = encoder.similarity(embeddings[i], embedded_ans).item()
        distances_to_ans.append(toans)
        cur = []
        for j in range(len(embeddings)):
            if j != i:
                tocur = 1.0 - encoder.similarity(embeddings[i], embeddings[j]).item()
                cur.append(tocur)
            else:
                cur.append(0.0)
        distances.append(cur)
    return distances, distances_to_ans

if __name__ == "__main__":
    dists, _ = get_distance_matrix(text_outputs, baseline_ans)
    f, (fig1, fig2) = plt.subplots(1, 2)
    fig1.bar(x= [output["model"] for output in text_outputs], height=_)
    fig1.tick_params(axis='x', rotation=45)
    plt.sca(fig2)
    fig2 = clustering.cluster_result(dists)
    drift_results = get_semantic_drift(tst_model=model, given_tokenizer=tokenizer, outputs=text_outputs, ans=baseline_ans)
    drift_results["figure"].show()
    anotherfig, axs = plt.subplots(1, len(drift_results["drifts"]), squeeze=False)
    b: int = 0
    for whatever in drift_results["drifts"]:
        plt.sca(axs[0, b])
        gramschmidt_process(whatever[2].cpu().numpy().tolist(), device=device)
    plt.show()
