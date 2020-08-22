---
layout: post
title:  "The Consciousness Prior"
category: review
date:   2020-08-17 14:06:00 -0600
categories: blog
cover-photo: assets/images/concprior-header.png
cover-photo-alt: "Consciousness Prior Header Photo"
icon: fa-flask
---

Recent psychology suggests that what we think of as conscious thought is may simply be selective attention paid to certain (mental) objects, with many other things being ignored by our conscious mind.
Yoshua Bengio's "consciousness prior" attempts to turn this psychological principle in to a useful machine learning principle, by using a subset of the available information to model dynamics using a sparse factor graph.

----------------------------
<br/>

## Authors
 - **Yoshua Bengio** *(Prof., UMontreal - MILA)*

 
***ArXiV***: *[https://arxiv.org/abs/1709.08568.pdf](https://arxiv.org/abs/1709.08568.pdf)*
 
## Background

In recent discussions of neuroscience, the concept of consciousness being intrinsically linked is becoming increasingly popular.
To clarify, here, we're not talking about subjective experience (like qualia) or free-will or anything, solely the idea of what you are 'aware of' at any given moment.
Bengio relates this to Daniel Kahnemann's concepts of 'System 1' and 'System 2' from his book 'Thinking, Fast and Slow' (which you should totally read by the way).
System 1 handles subconscious computations that are tough to define: instantly recoiling from heat or recognizing an apple as an apple.
System 2 involves much slower thought processes ... like deciding how to phrase a sentence explaining system 2.
This system is usually what we associate with our conscious self, because it's what we use to make rational and 'conscious' decisions.

Bengio points out that a lot of what deep learning does right now looks *a lot* like system 1 stuff.
I would personally add that a lot of old symbolic AI looks a lot like system 2.
Either way, Bengio argues that it may be useful to have a secondary system that's able to do more in-depth calculations on fewer variables.

## Core Idea

This paper essentially argues for a 3-step process for implementing this.
 1. *Extracting a subconscious state*: Bengio begins by assuming that we have some high-level representation of the world $$h_{t-1}$$ and some observation $$x_t$$. He recommends that your new representation of the world $$h_{t}$$ should be a function of your previous representation and your observation $$F(x_t, h_{t-1})$$. Because this sounds a lot like an RNN, he dubs this the representation RNN, with $$h_t$$ being the unconscious representation state.
 2. *Exttracting a conscious state*: If $$h_t$$ is composed of a set of elements, then we can select a subset of it to form the conscious state. Called $$c_t$$, he suggests that the update rule combines the unconscious representation, your conscious state, some memory and some randomness. He calls this process the *consciousness process*. The idea is that $$c_t$$ is probably pretty small, so we can afford to do more expensive computations on it.
 3. *Remembering the past*: The author suggests committing conscious states to some form of memory, which may or may not be parametric.
 4. *Factor Graphs*: Once we have a limited number of variables in our consciousness, we can run something like a sparse factor graph on it.
 
If $$S$$ is our conscious state consisting of elements $$\{S_1, S_2, ..., S_N\}$$, a factor graph defines:

$$P(S) = \frac{\prod_i f_i(S_i)}{Z}$$

where $$Z$$ is just that normalization constant you add to make things a valid probability distribution and $$f_i$$ is a function of the other elements, called the potential function.
The idea here is that each $$f_i$$ relies on a fairly limited number of other variables, so that the interaction graph is sparse.
Ideally, this graph includes the fewest variables necessary to make fairly strong predictions.

Bengio draws an analogy to language, where we don't describe everything in complete, precise detail.
Instead, we include the pieces of information that we need to get across *most* of the information.
Alternately, if you want to understand $$P$$ as an energy function, we include just the variables that are able to effectively decrease the energy of the system.

The 5th component is the training objective.
Bengio suggests a verifier network $$V$$.
If we want $$c_t$$ to have predictive power, then our conscious model should effectively predict future perceptions.
As a result, Bengio recommends taking some previous perceptual input $$h_t$$ and some conscious state that occurred prior to that perception $$c_{t-k}$$ and running some verification function on that.
He suggests several possible functions for this job, ranging from raw prediction quality e.g. maximizing $$\text{log} |P(h_t|c_{t-k})|$$ or using a proxy like a conditional GAN.

## Details & Experiments

You don't need experiments to get your thoughts out there when you're as big as Bengio.
In all seriousness, this paper is somewhat similar to what Jurgen Schmidhuber often writes, where it's hypothesizing on a big idea as a future direction.
It's pretty abstract, which you could read as leaving a lot of ground for people to write implementations of the idea.

## Future Directions & General Thoughts

While I think that this paper is pretty vague about what the 'consciousness prior' should look like in practice, I actually pretty strongly agree that this is part of the future of ML.
Although I wasn't around to live through it, my understanding of AI history is that one of the major shortcomings of the old symbolic model was computational complexity : a lot of the algorithm ran in $$O(n^2)$$ time or worse.
I think that selective consciousness to parts of the problem could give us an avenue for probabilistically solving these problems much faster on average.
What's more, having a limited attention can help explicability, which is a major concern for both AI ethics and AI safety going forward (as well as just being useful for engineering).
There were other problems (such as brittleness, which, granted, is still a problem), but I think that they can compliment each other's weaknesses well.

That being said, I don't think that we're close to implementing anything that has the full spirit of Bengio's consciousness prior in practice.
You could try a brute force method by using RIMs (as discussed in my previous post [here](https://blumx116.github.io/2020/08/15/rim.html)) for the unconscious state.
From here, you could then use attention to select from the outputs of the RIMs and then ... I'm not sure what to put on top.
A sparse factor model might work well for some modelling problems, but you still have to come up with each of those individual models.
Fundamentally, I think this is kind of the problem that we haven't figured out with deep learning here : we don't know how to combine deep learning with other methods in a complementary way.
The methods I've seen so far either seem too hand-coded (like [Neuro Symbolic VQA](https://blumx116.github.io/blog/review/2020/08/09/vqa.html)) or don't scale well (like [Neural Turing Machines](https://arxiv.org/pdf/1410.5401.pdf) utilizing external memory).

I have a lot of half-ideas here, but I think the ones that are closest to working are similar to the various causal models popping up in deep learning.
Overall, I like the framing and direction that this manuscript provides!
