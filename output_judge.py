import sklearn
import transformers
from sentence_transformers import SentenceTransformer, SimilarityFunction
import torch
import clustering
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")
print(f"using {device} as performance accelerator")
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device, similarity_fn_name=SimilarityFunction.COSINE)
model = transformers.AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = transformers.AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

model = model.to(device=device)
# tokenizer = tokenizer.to(device=device)

text_outputs: list[dict] = [
    # --- CLUSTER A: High-Accuracy / Semantically Identical ---
    {
        "model": "model_alpha_accurate",
        "text": "The capital recovery factor determines the equal annual payments needed to pay back an initial investment value over a set timeframe and interest rate."
    },
    {
        "model": "model_beta_accurate",
        "text": "To find the uniform series of end-of-period cash flows necessary to recover a principal investment cost over n periods at i interest, you use the capital recovery factor."
    },
    
    # --- MODERATELY CLOSE: Right domain, slightly drifting concept ---
    {
        "model": "model_gamma_partial",
        "text": "Capital recovery involves calculating annualized values for investments, ensuring that the initial capital put into a project is fully recovered over its economic life."
    },

    # --- CLUSTER B: Coherent but Completely Wrong (Different Economic Concept) ---
    {
        "model": "model_delta_wrong",
        "text": "The Internal Rate of Return is the specific discount rate that makes the net present value of all cash flows from a particular project equal to zero."
    },
    {
        "model": "model_epsilon_wrong",
        "text": "Net Present Value is calculated by subtracting the initial investment costs from the total present value of incoming cash flows over a given period."
    },

    # --- THE OUTLIER: Total System Failure / Crash Garbage ---
    {
        "model": "model_zeta_broken",
        "text": "Error 502: Bad Gateway. Connection timed out while requesting stream from OpenRouter upstream provider upstream_id_0x4f12a."
    }
]
baseline_ans: str = "The Capital Recovery Factor is used to calculate the uniform series of end-of-period payments required to recover an initial capital investment over a specified number of periods at a given interest rate."

def get_semantic_drift(tst_model, given_tokenizer, outputs: list[dict], ans: str):
    activation_storage: dict = {}
    target_layer = tst_model.encoder.layer[5]
    model_names = [output['model'] for output in outputs]
    def callback_function(module, in_tensor, out_tensor, model_name: str):
        activation_storage[model_name] = out_tensor.detach().cpu()
    # This is a placeholder we will fill in the PyTorch loop
    curmodel: str = ""

    with torch.no_grad():
        for candidate in outputs:
            curmodel = candidate["model"]
            hook = target_layer.register_forward_hook(lambda m, i, o, name=curmodel: callback_function(m, i, o, name))
            tokenized = given_tokenizer(candidate["text"], return_tensors="pt")
            tokenized = tokenized.to(device=device)
            _ = tst_model(**tokenized)
            hook.remove()
            del _
        curmodel = "answer"
        anshook = target_layer.register_forward_hook(lambda m, i, o, name=curmodel: callback_function(m, i, o, name))
        ans_tokenized = given_tokenizer(ans, return_tensors="pt")
        ans_tokenized = ans_tokenized.to(device=device)
        _ = tst_model(**ans_tokenized)
        anshook.remove()
        del _
    
    activation_storage["answer"] = activation_storage["answer"].squeeze(0)
    activation_storage["answer"] = torch.nn.functional.normalize(activation_storage["answer"])
    final_drifts: list[tuple[str, list]] = []
    for name, data in (activation_storage).items():
        if name == "answer": 
            continue
        else:
            data = data.squeeze(0)
            data = torch.nn.functional.normalize(data)
            final = data @ activation_storage["answer"].T
            final = final.tolist()
            final_drifts.append((name, final))
    fig, axs = plt.subplots(ncols=len(final_drifts), sharey=True, layout="constrained", nrows=1, squeeze=False)
    fig.suptitle(f"Semantic Drift From the Baseline Answer of {len(final_drifts)} Models")
    for i in range(len(final_drifts)):
        axs[0, i].imshow(final_drifts[i][1], cmap="viridis", vmin=0.7, vmax=1.0, aspect="equal")
        axs[0, i].set_title(final_drifts[i][0])
        axs[0, i].set_xlabel("Response")
        if i == 0:
            axs[0, i].set_ylabel("Baseline")
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
    plt.show()
