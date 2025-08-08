# Leaf

An open source "prototype" AI model used for AI research.

## About this project

Leaf is an "experimental" AI model, utilising PyTorch.

## Research

With leaf we've been testing many capabilities of what AI could do. 

Starting with a simple "embedded" python dataset, leaf uses only 2700 steps for training (the more steps, the  better it learns).

**Training Data:** `
{"this is a much longer text that will serve as a simple dataset for our tiny language model. The model will learn to predict the next character based on the previous characters in the sequence."}
{"text": "This demonstrates the core idea behind training an autoregressive language model. The quick brown fox jumps over the lazy dog."}
{"text": "A journey of a thousand miles begins with a single step. The early bird catches the worm. All that glitters is not gold. A stitch in time saves nine."}
{"text": "Where there's a will, there's a way. Look before you leap. You can't make an omelette without breaking a few eggs. Practice makes perfect. Don't count your chickens before they hatch."}`

However this result came with the following output: 

`text that will serve`

Then we used JSONL databases from the community, and unfortunatly this was the output:

`rimetricE7tich then`