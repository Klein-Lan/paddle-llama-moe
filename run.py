from model.model_llama_moe import LlamaForCausalLM
from model.config_llama_moe import LlamaConfig
from model.tokenizer import Tokenizer


import paddle


tokenizer = Tokenizer("tokenizer_models/llama-moe.model")
config = LlamaConfig(
    vocab_size=32000, 
    hidden_size=64,
    max_position_embeddings=762,
    intermediate_size=1000,
    num_hidden_layers=1,
    num_attention_heads=8,
)
input_ids = paddle.randint(0, 30000, [4, 30])
model = LlamaForCausalLM(config)
output = model(input_ids)
optim = paddle.optimizer.SGD(parameters=model.parameters())
print(model)

#Test Train
for step in range(1, 100):
    output = model.forward(paddle.randint(0, 30000, [4, 30]))[0]
    loss = output.mean()
    loss.backward()
    optim.step()
    optim.clear_grad()
    print("=== step : {}, loss : {}".format(step, loss.numpy()))




