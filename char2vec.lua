--[[
Class for char2vec with skipgram and negative sampling
--]]

require("sys")
require("nn")

local Char2Vec = torch.class("Char2Vec")

function Char2Vec:__init(config)
    self.tensortype = torch.getdefaulttensortype()
    self.neg_samples = config.neg_samples
    self.minfreq = config.minfreq
    self.minfreq_char = 2
    self.dim = config.dim -- dimension of word embeddings and char embeddings
    self.dim2 = config.dim2 -- dimension of char embeddings for softmax
    self.w2v_criterion = nn.BCECriterion() -- logistic loss
    self.word = torch.Tensor(1)
    self.contexts = torch.Tensor(1+self.neg_samples)
    self.labels = torch.zeros(1+self.neg_samples); self.labels[1] = 1 -- first label is always pos sample
    self.window = config.window
    self.lr = config.lr
    self.alpha = config.alpha
    self.lambda1 = config.lambda1
    self.lambda2 = config.lambda2
    self.lambda3 = config.lambda3
    self.table_size = config.table_size
    self.vocab = {} -- word counts
    self.index2word = {}; self.word2index = {}
    self.char_count = {} -- char counts
    self.char2index = {}; self.index2char = {}
    self.word2char = {}
    self.word2char2index = {}
    self.losses = {} -- track total l2 error, running avg of l2 error, and running avg of logistic loss
    self.pos = {"beg", "mid", "end", "all"}
    self.overlap_samples = config.overlap_samples -- number of overlap samples
    self.dl = torch.zeros(self.dim2)
    self.dp = torch.zeros(self.dim2)
end

-- Build vocab frequency, word2index, and index2word from input file
function Char2Vec:build_vocab(corpus)
    print("Building vocabulary of words...")
    local start = sys.clock()
    local total_count = 0
    local f = io.open(corpus, "r")
    local n = 1
    for line in f:lines() do
        for _, word in ipairs(self:split(line)) do
	    total_count = total_count + 1
	    if self.vocab[word] == nil then
	        self.vocab[word] = 1
            else
	        self.vocab[word] = self.vocab[word] + 1
	    end
        end
        n = n + 1
    end
    f:close()
    -- Delete words that do not meet the minfreq threshold and create word indices
    for word, count in pairs(self.vocab) do
    	if count >= self.minfreq then
     	    self.index2word[#self.index2word+1] = word
            self.word2index[word] = #self.index2word
    	else
	    self.vocab[word] = nil
        end
    end
    self.vocab_size = #self.index2word
    print(string.format("%d words and %d sentences processed in %.2f seconds.", total_count, n, sys.clock() - start))
    print(string.format("Vocab size after eliminating words occuring less than %d times: %d", self.minfreq, self.vocab_size))
    -- initialize word/context embeddings and the word2vec model
    self.word_vecs = nn.LookupTable(self.vocab_size, self.dim) -- context embeddings
    self.context_vecs = nn.LookupTable(self.vocab_size, self.dim)-- word embeddings
    self.w2v = nn.Sequential()
    self.w2v:add(nn.ParallelTable())
    self.w2v.modules[1]:add(self.context_vecs)
    self.w2v.modules[1]:add(self.word_vecs)
    self.w2v:add(nn.MM(false, true)) -- dot prod and sigmoid to get probabilities
    self.w2v:add(nn.Sigmoid())
    -- N(0,1) is too large, so make it smaller
    self.word_vecs:reset(0.25)
    self.context_vecs:reset(0.25)
end

-- Build dictionary of word to char-ngram mappings and also create char2index/index2char mappings
function Char2Vec:build_chars()
    print("Building vocabulary of character ngrams...")
    local start = sys.clock()
    -- get character ngram counts and word to char mappings
    for word, _ in pairs(self.vocab) do
        local chars = self:get_char_ngrams(word, 1, word:len())
	self.word2char[word] = chars
	for char, char_cnt in pairs(chars) do
	    if self.char_count[char]==nil then
	        self.char_count[char] = #char_cnt
	    else
                self.char_count[char] = self.char_count[char] + #char_cnt
	    end
	end
    end
    -- remove character ngrams that occur less than < minfreq_char
    for char, _ in pairs(self.char_count) do
        if self.char_count[char] >= self.minfreq_char then
	    self.index2char[#self.index2char+1] = char
	    self.char2index[char] = #self.index2char
	else
	    self.char_count[char] = nil
	end
    end
    -- get word to character mappings for those that meet minfreq_char threshold
    local max_len = 0; local max_pos = 0
    for word, chars in pairs(self.vocab) do
        local char_features = self:get_char_features(word) -- char features for a given word
	self.word2char2index[self.word2index[word]] = char_features--, char_features[2]}
    end
    print(string.format("Done in %.2f seconds.", sys.clock() - start))
    print(string.format("Char-ngram vocab size after eliminating char-ngrams occuring less than %d times: %d",
            self.minfreq_char, #self.index2char))
    -- character ngram embeddings
    self.char_vecs = nn.LookupTable(#self.index2char, self.dim2) -- char embeddings for softmax
    self.char_vecs2 = nn.LookupTable(#self.index2char, self.dim) -- char embeddings to be added to make word embeddings
    -- concatenation layer that concantenates char-embeddings and the features to be fed into softmax
    self.concat = nn.ParallelTable()
    self.concat:add(self.char_vecs) -- char embeddings for softmax
    --self.concat:add(nn.LookupTable(4, 4)) -- position features (begin, middle, end, wholeword)
    self.concat.modules[1]:reset(0.01); --self.concat.modules[2]:reset(0.01)
    self.logistic = nn.Sequential()
    self.logistic:add(self.concat)
    self.logistic:add(nn.JoinTable(2,2))
    self.logistic:add(nn.Linear(self.dim2,1))
    local concat2 = nn.ParallelTable()
    concat2:add(self.char_vecs2) -- char embeddings to be added
    concat2.modules[1]:reset(0.25)
    -- create softmax layer to get weights for char embeddings
    self.mlp = nn.Sequential()
    self.mlp:add(self.logistic)
    self.mlp:add(nn.Transpose({2,1}))
    self.mlp:add(nn.SoftMax())
    -- word embedding is a weighted average of char embeddings
    self.c2v = nn.Sequential()
    self.c2v:add(nn.ConcatTable())
    self.c2v.modules[1]:add(self.mlp)
    self.c2v.modules[1]:add(nn.Sequential())
    self.c2v.modules[1].modules[2]:add(concat2)
    self.c2v.modules[1].modules[2]:add(nn.SelectTable(1)) -- only get the char indices
    self.c2v:add(nn.MM())
end

-- Build a table of unigram frequencies from which to obtain negative samples
function Char2Vec:build_table()
    local start = sys.clock()
    local total_count_pow = 0
    print("Building a table of unigram frequencies... ")
    for _, count in pairs(self.vocab) do
    	total_count_pow = total_count_pow + count^self.alpha
    end
    self.table = torch.IntTensor(self.table_size)
    local word_index = 1
    local word_prob = self.vocab[self.index2word[word_index]]^self.alpha / total_count_pow
    for idx = 1, self.table_size do
        self.table[idx] = word_index
        if idx / self.table_size > word_prob then
	    word_index = word_index + 1
	    if self.index2word[word_index] ~= nil then
	        word_prob = word_prob + self.vocab[self.index2word[word_index]]^self.alpha / total_count_pow
	    end
        end
        if word_index > self.vocab_size then
            word_index = word_index - 1
        end
    end
    print(string.format("Done in %.2f seconds.", sys.clock() - start))
end

-- Train on word context pairs using char2vec loss
function Char2Vec:train_pair_c2v(word, contexts)
    -- fwd prop on w2v model
    local p = self.w2v:forward({contexts, word})
    local log_loss = self.w2v_criterion:forward(p, self.labels)
    -- fwd prop on c2v model
    local char_features_all = self.word2char2index[word[1]]
    local char_features = {char_features_all[1]}
    if char_features[1]:dim() > 0 then
       local num_char_features = char_features[1]:size()[1]
       local q = self.c2v:forward(char_features) -- convex comb. of char embeddings
       local w = self.w2v.modules[1].modules[2].output -- current word embedding
       local diff = w-q -- this is dl_dq portion of (w-q)^2
       local l2_loss = torch.norm(diff)
       diff:mul(-self.lambda1)

       -- backpropagate errors from w2v loss
       dl_dp = self.w2v_criterion:backward(p, self.labels)
       self.w2v:zeroGradParameters()
       self.w2v:backward({contexts, word}, dl_dp)
       self.w2v:updateParameters(self.lr)

       -- backpropagate errors from l2 loss between w and q
       self.c2v:zeroGradParameters()
       self.c2v:backward(char_features, diff)
       self.c2v:updateParameters(self.lr)
       self.w2v.modules[1].modules[2].weight[word[1]]:add(diff:mul(self.lr))

       --backpropagate errors from overlap loss
       if num_char_features > 1 then
	   local num_samples = math.min(num_char_features, self.overlap_samples) -- num samples (default 2)
	   local char_weights= self.mlp.output:view(num_char_features) -- weights of the softmax
	   local char_sample = torch.multinomial(char_weights, num_samples, false):view(num_samples)
	   for i = 1, num_samples-1 do -- check every two-way chargram for overlaps
	       for j = 2, num_samples do
		   overlap1, overlap2, char_idx_i, char_idx_j = self:find_overlap(char_sample[i], char_sample[j],
			   char_features_all, self.index2word[word[1]])
		   if overlap1 > 0 then -- there is at least one overlap
		       self.dp:mul(0) -- zero out gradients
		       for k = 1, num_char_features do
			   local l = char_features[1][k]
			   local d_i = 0; local d_j = 0 -- kronecker delta(s)
			   if l==char_idx_i then
			       d_i = 1
			   else
			       self.dp:add((self.char_vecs.weight[char_idx_i]-self.char_vecs.weight[l]):mul(char_weights[i]*char_weights[k]*overlap1))
			   end
			   if l==char_idx_j then
			       d_j = 1
			   else
			       self.dp:add((self.char_vecs.weight[char_idx_j]-self.char_vecs.weight[l]):mul(char_weights[j]*char_weights[k]*overlap2))
			   end
			   local a = overlap1*(char_weights[i]*(d_i - char_weights[k])) + overlap2*(char_weights[j]*(d_j - char_weights[k]))
			   self.dl:mul(self.logistic.modules[3].weight, -a*self.lr*self.lambda2)
			   self.char_vecs.weight[l]:add(self.dl) -- update char embeddings
		       end
		       self.dp:mul(-self.lr*self.lambda2) -- update linear weights
		       self.logistic.modules[3].weight:add(self.dp)
		   end
	       end
	   end
       end
       -- l2 penalty on char embeddings that are fed into the softmax
       for i = 1, num_char_features do
	   k = char_features[1][i]
	   l = -self.char_vecs.weight[k]
	   self.char_vecs.weight[k]:add(l:mul(self.lr*self.lambda3))
       end
    return log_loss, l2_loss

    else
        return log_loss, 0
    end
end

-- Sample negative contexts
function Char2Vec:sample_contexts(context)
    self.contexts[1] = context
    local i = 0
    while i < self.neg_samples do
        neg_context = self.table[torch.random(self.table_size)]
	if context ~= neg_context then
	    self.contexts[i+2] = neg_context
	    i = i + 1
	end
    end
end

-- Train on sentences
function Char2Vec:train_model(corpus)
    print("Training...")
    --local test_words = {"the","and","called", "however", "because", "government", "international", "subdivide","nonpolitical"}
    local test_words = {"#the#","#and#","#called#", "#however#", "#because#", "#government#", "#international#", "#subdivide#","#nonpolitical#"}
    local start = sys.clock()
    local c = 0
    local log_loss = 0
    local l2_loss = 0
    f = io.open(corpus, "r")
    for line in f:lines() do
        sentence = self:split(line)
        for i, word in ipairs(sentence) do
	    word_idx = self.word2index[word]
	    if word_idx ~= nil then -- word exists in vocab
    	        local reduced_window = torch.random(self.window) -- pick random window size
		self.word[1] = word_idx -- update current word
                for j = i - reduced_window, i + reduced_window do -- loop through contexts
	            local context = sentence[j]
		    if context ~= nil and j ~= i then -- possible context
		        context_idx = self.word2index[context]
			if context_idx ~= nil then -- valid context
  		            self:sample_contexts(context_idx) -- update pos/neg contexts
			    ll, l2 = self:train_pair_c2v(self.word, self.contexts)
			    log_loss = log_loss + ll -- train word context pair
			    l2_loss = l2_loss + l2
			    c = c + 1
			    if c % 100000 ==0 then
			        print(string.format("%d words trained in %.2f seconds with lr: %.8f", c, sys.clock() - start, self.lr))
				local error = self:check_distance()
				self.losses[#self.losses + 1] = {l2_loss/c, log_loss/c, error, c}
				print(string.format("L2 error (total): %.2f, L2 error (avg): %.4f, Logistic loss: %.4f",
				        error, l2_loss/c, log_loss/c))
			        self:print_sim_words(test_words, 4)
				for _, x in pairs(test_words) do
				    print("--------------------")
				    self:print_char_weights(x, 8)
				end
				self.word_vecs_norm = nil
				self.word_char_vecs = nil
			    end
			end
		    end
	        end
	    end
	end
    end
end

-- prints chargram weights (e.g. the = 0.1*t + 0.2*th ...)
function Char2Vec:print_char_weights(word, k)
    local char_features = self:get_char_features(word)
    local eq = {}
    local w = self.c2v.modules[1]:forward(char_features)
    _, idx = torch.sort(-w[1])
    local k = k or idx:size(2)
    for i = 1, math.min(k, idx:size(2)) do
        local pos_i = char_features[2][idx[1][i]]
	local idx_i = char_features[1][idx[1][i]]
        eq[i] = string.format("%.3f * %s (%s)", w[1][1][idx[1][i]], self:get_char(idx_i), self.pos[pos_i])
    end
    print(word .. " = \n" .. table.concat(eq, " + \n"))
end

-- obtain word embeddings from chargram embeddings
function Char2Vec:get_word_char_vecs()
    self.word_char_vecs = torch.randn(self.vocab_size, self.dim)
    for i, _ in ipairs(self.index2word) do
        self.word_char_vecs[i] = self.c2v:forward(self.word2char2index[i])
    end
end

-- check l2 distance btw word embeddings and char embeddings
function Char2Vec:check_distance()
    if self.word_char_vecs == nil then
        self:get_word_char_vecs()
    end
    return torch.norm(self.word_char_vecs - self.w2v.modules[1].modules[2].weight:clone())
end


-- Row-normalize a matrix
function Char2Vec:normalize(m)
    local m_norm = torch.zeros(m:size())
    for i = 1, m:size(1) do
    	m_norm[i] = m[i] / torch.norm(m[i])
    end
    return m_norm
end

-- Return the k-nearest words to a word or a vector based on cosine similarity
-- w can be a string such as "king" or a vector for ("king" - "queen" + "man")
function Char2Vec:get_sim_words(w, k)
    if self.word_vecs_norm == nil then
        self.word_vecs_norm = self:normalize(self.w2v.modules[1].modules[2].weight)
    end
    if type(w) == "string" then
        w = self.word_vecs_norm[self.word2index[w]] or self.c2v:forward(self:get_char_features(w)):view(self.dim)
    end
    local sim = torch.mv(self.word_vecs_norm, w)
    sim, idx = torch.sort(-sim)
    local r = {}
    for i = 1, k do
        r[i] = {self.index2word[idx[i]], -sim[i]}
    end
    return r
end

function Char2Vec:print_sim_words(words, k)
    for i = 1, #words do
    	r = self:get_sim_words(words[i], k)
	if r ~= nil then
   	    print("-------"..words[i].."-------")
	    for j = 1, k do
	        print(string.format("%s, %.4f", r[j][1], r[j][2]))
	    end
	end
    end
end

-- Get all character ngrams given ngram range
function Char2Vec:get_char_ngrams(word, low, high)
    local ngrams = {}
    local word_len = word:len()
    for i = low, high do
        self:get_char_ngram(word, word_len, i, ngrams)
    end
    return ngrams
end

-- Get all character ngrams of length n
function Char2Vec:get_char_ngram(word, word_len, n, ngrams)
    for i = 1, word_len-n+1 do
    	ngram = word:sub(i, i+n-1)
	if ngrams[ngram]==nil then
	    ngrams[ngram] = {i}
	else
	    ngrams[ngram][#ngrams[ngram]+1] = i
	end
    end
end

-- Get chargram features for a given word
function Char2Vec:get_char_features(word)
    local word_l = word:len()
    local chars = self:get_char_ngrams(word, 1, word_l)
    local char_i = {} -- char index
    local pos_i = {} -- pos index (beg, mid, end, all)
    local pos2_i = {} -- pos index2 (beginning position)
    local len_i = {} -- length
    local i = 0
    for char, pos in pairs(chars) do
        local l = char:len()
	if self.char_count[char] ~= nil then
	    for _, p in ipairs(pos) do
	    	--if l==word_l or p==1 or p+l-1==word_l or l > 1 then
		if l > 1 and p ~= 2 then
	        i = i + 1
		len_i[i] = l
		if l==word_l then
		    pos_i[i] = 4 -- whole word
		elseif p == 1 then
		    pos_i[i] = 1 -- beginning
		elseif p+l-1 == word_l then
		    pos_i[i] = 3 -- end
		else
		    pos_i[i] = 2 -- middle
		end
		--char_i[i] = self.char2index[char] + (4-pos_i[i])*#self.index2char
		char_i[i] = self.char2index[char]
		pos2_i[i] = p
		end
	    end
	end
    end
    return {torch.Tensor(char_i),  torch.Tensor(pos_i), torch.Tensor(pos2_i), torch.Tensor(len_i)}
end

-- split on separator
function Char2Vec:split(input, sep)
    if sep == nil then
        sep = "%s"
    end
    local t = {}; local i = 1
    for str in string.gmatch(input, "([^"..sep.."]+)") do
        t[i] = str; i = i + 1
    end
    return t
end

-- find overlap of two chargrams given char_features and indices
function Char2Vec:find_overlap(i,j,char_features,word)
    local overlap = 0
    local char1 = char_features[1][i]; local char2 = char_features[1][j]
    local pos1 = char_features[3][i]; local pos2 = char_features[3][j]
    local len1 = char_features[4][i]; local len2 = char_features[4][j]
    local pos1_end = pos1 + len1 - 1; local pos2_end = pos2 + len2 - 1
    if pos1 <= pos2 and pos1_end >= pos2 then
        overlap = len2 - math.max(0, pos2_end - pos1_end)
    elseif pos2 < pos1 and pos2_end >= pos1 then
	overlap = len1 - math.max(0, pos1_end - pos2_end)
    end
    local l = math.abs(word:len() - (len1+len2))
    local overlap1 = overlap/len1 --normalize so we don't penalize longer ngrams
    local overlap2 = overlap/len2
    --overlap = overlap + math.abs(word:len() - (len1+len2))
    return overlap1, overlap2, char1, char2
end

-- returns chargram given an index
function Char2Vec:get_char(char_idx)
    while self.index2char[char_idx]==nil do
        char_idx = char_idx - #self.index2char
    end
    return self.index2char[char_idx]
end
