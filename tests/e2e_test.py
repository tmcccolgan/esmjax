import numpy as np
from flax.core import frozen_dict

from esmjax import io
from esmjax import tokenizer as esm_tokenizer
from esmjax.modules import modules

MODEL_NAME = "esm2_t6_8M_UR50D"
# Load in the original PyTorch state; will download if first time.
state = io.get_torch_state(MODEL_NAME)

esm, params_axes = modules.get_esm2_model(state["cfg"])
esm_params = io.convert_encoder(state["model"], state["cfg"])
esm_params = frozen_dict.FrozenDict({"params": esm_params})


p53_seq = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP\
    DEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAK\
    SVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHE\
    RCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNS\
    SCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELP\
    PGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPG\
    GSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD"

insulin_seq = "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED\
    LQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN"

tokenizer = esm_tokenizer.protein_tokenizer(pad_to_multiple_of=128)
tokens = [x.ids for x in tokenizer.encode_batch([p53_seq, insulin_seq])]
batch = np.array(tokens)

print(batch)

# calculate embeddings by calling apply on esm

embeddings = esm.apply(esm_params, batch)

print(embeddings)
