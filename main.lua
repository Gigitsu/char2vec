--[[
Code to process text data.
Input is a text file where each line is a sentence.
--]]

require("io")
require("os")
require("paths")
require("torch")
dofile("char2vec.lua")

-- Default configuration
config = {}
config.corpus = "sentences2.txt" -- input data
config.window = 3 -- (maximum) window size
config.dim = 100 -- dimensionality of word/char embeddings
config.dim2 = 50 -- dimensionality of char embeddings for softmax
config.alpha = 0.75 -- smooth out unigram frequencies
config.table_size = 1e6 -- table size from which to sample neg samples
config.neg_samples = 5 -- number of negative samples for each positive sample
config.minfreq = 100 --threshold for vocab frequency
config.lr = 0.025 -- initial learning rate
config.lambda1 = 0.05 -- penalty of l2 loss between word/char embeddings
config.lambda2 = 0.01 -- penalty for overlap loss
config.lambda3 = 2.0 -- l2 penalty for softmax char embeddings
config.overlap_samples = 2 -- number of chargrams to sample

-- Parse input arguments
cmd = torch.CmdLine()
cmd:option("-corpus", config.corpus)
cmd:option("-window", config.window)
cmd:option("-minfreq", config.minfreq)
params = cmd:parse(arg)
for param, value in pairs(params) do
    config[param] = value
end

model = Char2Vec(config)
model:build_vocab(config.corpus)
model:build_chars()
model:build_table()
model:train_model(config.corpus)
