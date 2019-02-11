# PCFG dataset

The PFCG dataset is obtained from the random split obtained by Mathijs Mul.  Please find the scripts used to generate this dataset at [https://github.com/MathijsMul/pcfg-set](https://github.com/MathijsMul/pcfg-set). 


### Example

The dataset comprises of random sequences of tokens and specific operations that can be applied to these sequences. For instance the operation `shift` will shift the first element of the sequence to the back of the sequence:

Input:
```
shift E9 R13 I13 K9 K15
```

Output:
```
R13 I13 K9 K15 E9
```

All possible **unary** operations include:

- `copy` - returns the sequence
- `reverse` - reverses the order of the sequence
- `echo` - appends the last element of the sequence to the sequence
- `swap_first_last` - swaps the first and last elements
- `repeat` - repeats the whole sequence and appends it
- `shift` - removes the first element of the sequence and appends it to the end

There are also **binary** operations which can be applied to two sequences at once: 

- `append` - appends the second sequence to the first
- `prepend` - prepends the second sequence to the first
- `remove_first` - returns only the second sequence
- `remove_second` - returns only the first sequence
