Title: Python Code for Bayesian Filter Models
Date: 2018-08-04
Tags: datascience, bayesian statistics, machine learning
Slug: bayesianfilters
Author: Chris Gagne
Summary: In this post, I share some code that I used to build Bayesian filters.

In this post, I want to share the code that I used to build Bayesian filters. These models are commonly used in robotics (see [Peter Abbeel's lecture](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa13/slides/bayes-filters.pdf)). However, in neuroscience, we use these models to study how the brain represents different (higher order) levels of uncertainty. The two most relevant neuroscientific papers are [Behrens 2007](http://www.nature.com/neuro/journal/v10/n9/abs/nn1954.html), and [Boorman 2016](http://dx.doi.org/10.1016/j.neuron.2016.02.014). (The authors of these papers kindly shared their matlab code, which I used for my own pythonic implementation).

What are Bayesian filters? Abstractly, filters estimate observed data $x_t$ at a particular time point using observed data up to that time point $ x_0,...,x_{t-1}$. Often, the filters assume the existence of and try to estimate hidden states of the world $z_0, ..., z_{t-1}$ from the observations. The hidden states are usually assumed to be Markovian--the current state only depends on the previous one.

I'll discuss two alternative models for the hidden state dynamics and step through the code for one of them. First, I'll discuss a model in which the hidden state randomly switches between two values and then I'll discuss a hidden state that drifts as a random walk.

### The Switching Model

What I'll refer to as the 'switching model' assumes the underlying world consists of two states, randomly alternating between them. Within state 1, outcome A occurs with probability $q$, and outcome B occurs with probability $1-q$. These outcomes are the observed (Bernoulli) variables $x_t \in {0,1}$. Within state 2, the outcome contingencies switch; outcome A occurs with probability $1-q$ and B occurs with probability $q$.

We'll make the hidden state $z_t \in {0,1}$ whether or not a switch occured.    


The job of the filter model is to first estimate the unknown parameter $q_t$ on each trial

$ p(q_t | x_0, ..., x_t) $

and then to predict the next outcome

$ p(x_t | q_t) $

How do we do this? The key is to do *recursive estimation*; we start by applying Bayes rule to the first trial and then we use the posterior as the next prior, repeating indefinitely.


Let's import necessary packages. The code can all be found in the [repo](https://github.com/crgagne/crgagne.github.io/content/hierarchical_guassian_filters) for this blog.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
%matplotlib inline
from imp import reload

# import code
import models
reload(models)
from models import *

# this is for displaying the code from a file
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import IPython

def display_code(file,lines):
    file = './'+file
    with open(file) as f:
        code = f.read()
    code = '\n'.join(code.split('\n')[lines[0]:lines[1]])

    formatter = HtmlFormatter()
    return(IPython.display.HTML('<style type="text/css">{}</style>{}'.format(
        formatter.get_style_defs('.highlight'),
        highlight(code, PythonLexer(), formatter))))
```

#### t=0

Before we have observed anything, we specify a prior for our unknown parameter $p(q_0)$. For this implementation, since we only have 1 hidden parameter $q$, we can construct a uniform prior on possible $q$'s using a grid. (Note: that grid representations of probability densities break down quickly as the dimensionality of the parameter space increases, but we'll be fine for these models).

In the code printed below, I set up a grid for the prior. Even though our prior is 1-D, I'm making the matrix 3-D so that I don't have to broadcast it later when multiplying by other density functions. (note that the jupyter cell just prints code from 'models.py')


```python
display_code('models.py',[208,210])
```




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span>		<span class="c1"># create prior #</span>
		<span class="bp">self</span><span class="o">.</span><span class="n">prior_dist</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">ones</span><span class="p">((</span><span class="bp">self</span><span class="o">.</span><span class="n">grid_q</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">0</span><span class="p">],</span><span class="bp">self</span><span class="o">.</span><span class="n">grid_q</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">],</span><span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">outcomes</span><span class="p">)))</span>
</pre></div>




Now that we have our prior, we can observe the first observation and calculate our first posterior. We take the observed outcome (either a 1 or 0) and multiply it by each possible value for q (`grid_q`) to created our likelihood grid $p(y_t|q_t)$.


```python
display_code('models.py',[234,238])
```




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span>		<span class="k">for</span> <span class="n">trial</span><span class="p">,</span><span class="n">y</span> <span class="ow">in</span> <span class="nb">enumerate</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">outcomes</span><span class="p">):</span>

			<span class="n">lik</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">grid_q</span><span class="o">**</span><span class="p">(</span><span class="n">y</span><span class="p">)</span><span class="o">*</span><span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="bp">self</span><span class="o">.</span><span class="n">grid_q</span><span class="p">)</span><span class="o">**</span><span class="p">(</span><span class="mi">1</span><span class="o">-</span><span class="n">y</span><span class="p">)</span> <span class="c1"># bernouli distribution for likelihood</span>
</pre></div>




We multiply this likelihood by the prior to get the first posterior.

$$p(q_0|y_0)\propto p(y_0|q_0)p(q_0) $$


```python
display_code('models.py',[248,252])
```




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span>			<span class="c1"># mutliple prior and liklihood</span>
			<span class="bp">self</span><span class="o">.</span><span class="n">post_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">]</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">multiply</span><span class="p">(</span><span class="n">lik</span><span class="p">,</span><span class="bp">self</span><span class="o">.</span><span class="n">prior_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">])</span>
			<span class="bp">self</span><span class="o">.</span><span class="n">post_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">]</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">post_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">]</span><span class="o">/</span><span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">post_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">])</span> <span class="c1"># normalize</span>
</pre></div>




#### t=1
After the first observation, we also need to consider the probability that $q$ stays the same or switches, which we denote as $p(q_t|q_{t-1})$. Our posterior after two observations is now:

$$p(q_1|y_0,y_1)\propto p(y_1|q_1)p(q_1|q_0)p(y_0|q_0)p(q_0)$$

We can be simplified by using the posterior from the first (zero'th) observation:

$$p(q_1|y_0,y_1)\propto p(y_1|q_1)p(q_1|q_0)p(q_0|y_0)$$

This multiplication is done in the lines of code below. Note that the `trans_func` refers to  $p(q_t|q_{t-1})$, and that densities are normalized before each multiplication to aid with numerical stability (numbers not getting to small or too big with repeated multiplication).


```python
display_code('models.py',[246,261])
```




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span>	    <span class="k">if</span> <span class="n">trial</span><span class="o">&gt;</span><span class="mi">0</span><span class="p">:</span>
	    <span class="c1"># apply transition function from old posterior to new prior #</span>
		<span class="n">old_post</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">post_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="o">-</span><span class="mi">1</span><span class="p">]</span>
		<span class="n">old_post_unpacked</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">repeat</span><span class="p">(</span><span class="n">old_post</span><span class="p">[:,:,</span><span class="n">np</span><span class="o">.</span><span class="n">newaxis</span><span class="p">],</span><span class="nb">len</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">q_range</span><span class="p">),</span><span class="n">axis</span><span class="o">=</span><span class="mi">2</span><span class="p">)</span>
		<span class="c1"># dim are v,qt-1,qt, expanded old v and qt-1 into new qt</span>

		<span class="n">new_prior_unpacked</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">multiply</span><span class="p">(</span><span class="n">old_post_unpacked</span><span class="p">,</span><span class="bp">self</span><span class="o">.</span><span class="n">trans_func</span><span class="p">)</span>
		<span class="c1"># multiply old posterior by probaility that q stayed the same #</span>
		<span class="n">new_prior</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">new_prior_unpacked</span><span class="p">,</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span> <span class="c1"># sum out qt-1</span>
		<span class="bp">self</span><span class="o">.</span><span class="n">prior_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">]</span> <span class="o">=</span> <span class="n">new_prior</span><span class="o">/</span><span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="n">new_prior</span><span class="p">)</span> <span class="c1"># normalized again</span>

	    <span class="c1"># mutliple prior and liklihood</span>
	    <span class="bp">self</span><span class="o">.</span><span class="n">post_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">]</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">multiply</span><span class="p">(</span><span class="n">lik</span><span class="p">,</span><span class="bp">self</span><span class="o">.</span><span class="n">prior_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">])</span>
	    <span class="bp">self</span><span class="o">.</span><span class="n">post_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">]</span> <span class="o">=</span> <span class="bp">self</span><span class="o">.</span><span class="n">post_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">]</span><span class="o">/</span><span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">post_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">])</span> <span class="c1"># normalize</span>
</pre></div>




#### t>1
After the second observation, we repeat for all the observations up to time $t$ using the same procedure. After we are finished, we can get the posterior distribution for $q$ by marginalizing the whole posterior. This is what we'll visualize for the inference.


```python
display_code('models.py',[262,264])
```




<style type="text/css">.highlight .hll { background-color: #ffffcc }
.highlight  { background: #f8f8f8; }
.highlight .c { color: #408080; font-style: italic } /* Comment */
.highlight .err { border: 1px solid #FF0000 } /* Error */
.highlight .k { color: #008000; font-weight: bold } /* Keyword */
.highlight .o { color: #666666 } /* Operator */
.highlight .ch { color: #408080; font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: #408080; font-style: italic } /* Comment.Multiline */
.highlight .cp { color: #BC7A00 } /* Comment.Preproc */
.highlight .cpf { color: #408080; font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: #408080; font-style: italic } /* Comment.Single */
.highlight .cs { color: #408080; font-style: italic } /* Comment.Special */
.highlight .gd { color: #A00000 } /* Generic.Deleted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gr { color: #FF0000 } /* Generic.Error */
.highlight .gh { color: #000080; font-weight: bold } /* Generic.Heading */
.highlight .gi { color: #00A000 } /* Generic.Inserted */
.highlight .go { color: #888888 } /* Generic.Output */
.highlight .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
.highlight .gt { color: #0044DD } /* Generic.Traceback */
.highlight .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: #008000 } /* Keyword.Pseudo */
.highlight .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: #B00040 } /* Keyword.Type */
.highlight .m { color: #666666 } /* Literal.Number */
.highlight .s { color: #BA2121 } /* Literal.String */
.highlight .na { color: #7D9029 } /* Name.Attribute */
.highlight .nb { color: #008000 } /* Name.Builtin */
.highlight .nc { color: #0000FF; font-weight: bold } /* Name.Class */
.highlight .no { color: #880000 } /* Name.Constant */
.highlight .nd { color: #AA22FF } /* Name.Decorator */
.highlight .ni { color: #999999; font-weight: bold } /* Name.Entity */
.highlight .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
.highlight .nf { color: #0000FF } /* Name.Function */
.highlight .nl { color: #A0A000 } /* Name.Label */
.highlight .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
.highlight .nt { color: #008000; font-weight: bold } /* Name.Tag */
.highlight .nv { color: #19177C } /* Name.Variable */
.highlight .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
.highlight .w { color: #bbbbbb } /* Text.Whitespace */
.highlight .mb { color: #666666 } /* Literal.Number.Bin */
.highlight .mf { color: #666666 } /* Literal.Number.Float */
.highlight .mh { color: #666666 } /* Literal.Number.Hex */
.highlight .mi { color: #666666 } /* Literal.Number.Integer */
.highlight .mo { color: #666666 } /* Literal.Number.Oct */
.highlight .sa { color: #BA2121 } /* Literal.String.Affix */
.highlight .sb { color: #BA2121 } /* Literal.String.Backtick */
.highlight .sc { color: #BA2121 } /* Literal.String.Char */
.highlight .dl { color: #BA2121 } /* Literal.String.Delimiter */
.highlight .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
.highlight .s2 { color: #BA2121 } /* Literal.String.Double */
.highlight .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
.highlight .sh { color: #BA2121 } /* Literal.String.Heredoc */
.highlight .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
.highlight .sx { color: #008000 } /* Literal.String.Other */
.highlight .sr { color: #BB6688 } /* Literal.String.Regex */
.highlight .s1 { color: #BA2121 } /* Literal.String.Single */
.highlight .ss { color: #19177C } /* Literal.String.Symbol */
.highlight .bp { color: #008000 } /* Name.Builtin.Pseudo */
.highlight .fm { color: #0000FF } /* Name.Function.Magic */
.highlight .vc { color: #19177C } /* Name.Variable.Class */
.highlight .vg { color: #19177C } /* Name.Variable.Global */
.highlight .vi { color: #19177C } /* Name.Variable.Instance */
.highlight .vm { color: #19177C } /* Name.Variable.Magic */
.highlight .il { color: #666666 } /* Literal.Number.Integer.Long */</style><div class="highlight"><pre><span></span>	    <span class="bp">self</span><span class="o">.</span><span class="n">marg_v</span><span class="p">[:,</span><span class="n">trial</span><span class="p">]</span><span class="o">=</span><span class="n">np</span><span class="o">.</span><span class="n">sum</span><span class="p">(</span><span class="bp">self</span><span class="o">.</span><span class="n">post_dist</span><span class="p">[:,:,</span><span class="n">trial</span><span class="p">],</span><span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
</pre></div>




#### Running the code

Let's run the code from from start to finish. I first instantiate a `switch_model` object, then I load in some fake data, and then I run the inference code that we went through above.


```python
# get example subject
folder = os.getcwd()
filee = '/example_sub.csv'

# instantiate a model object
switch_model = model_switching(model_name='switching_w_uniform_prior',datafile_name=folder+filee)
switch_model.specify_model_subclass(0)
switch_model.load_data()
switch_model.initialize_parameter_range()
switch_model.initialize_transition()
switch_model.initialize_priors()

# do inference
switch_model.run_inference()
```

    180


After running the code, we can visualize the posterior $p(q_t|y_{0:t})$ for each time point. The color indicates higher probability, the y-axis is the possible value for $q$, the x-axis is time points, the red line is the true probability for $q_t$, which switches every 20 observations and then stays the same after observation #90, and the blue dots are the observations $x_t$.  

We can see that the model is doing a pretty good job at estimating $q_t$, but tends to overshoot the true value (the posterior mean tends to be above or below the red line). We can also see some false alarms after observation #90 when the posterior is higher for $q_t$ around 0.8 even though $q_t$ has not switched. Either way, pretty cool huh!


```python
sns.set(rc={'image.cmap': 'cubehelix'})
sns.set_context('talk')
fig,ax = plt.subplots(1,1,figsize=(12,8))
ylims = ax.get_ylim()
xlims = [0,180]

# plot outcome and true q
plt.plot(switch_model.data['probabilities'],color='r',label='true $q_t$')
x=switch_model.data['outcome']
plt.scatter(np.arange(len(x)),x,color='b',label='outcome $x_t\in{0,1}$')

# plot posterior
ax.imshow(switch_model.marg_q[:,0:-1],aspect='auto',extent=[xlims[0],xlims[1],ylims[0],ylims[1]],origin='lower')
ax.set_title('Posterior Distribution on $q$')
ax.set_xlabel('Observations')
ax.set_ylabel('Possible values for $q$')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
```




    <matplotlib.legend.Legend at 0x1a68544128>




![png](../../assets/images/Hierarchical_Bayesian_Filter_Models_files/Hierarchical_Bayesian_Filter_Models_20_1.png)


### Hierarchical Gaussian Model

Instead of assuming that $q$ switches with some fixed probability $p(q_t|q_{t-1})$, we could have assumed that it drifts after each observation by some amount. We'll call the average amount of this drift $v$ for variance, so that now our transition density is $p(q_{t}|q_{t-1},v_{t})$. Furthermore, we could assume that the variance in this drift also changes over time as well $p(v_{t}|v_{t-1},k)$. Then the posterior for $q$ and $v$ on each trial is then given by:


$$ p(q_{t},v_{t}|y_{1:t-1}) \propto p(y_{t}|q_{t}) \dots $$
$$ \int \int p(q_{t-1}, v_{t-1}, k_{t-1} | y_{1:t-1})p(v_{t}|v_{t-1},k) p(q_{t}|q_{t-1},v_{t}) dv dr $$

See [Behrens 2007](http://www.nature.com/neuro/journal/v10/n9/abs/nn1954.html) for more detail.

I wont go through the code line by line, but it's a very similar implementation to the 'switching model'. We'll just look at the results.


```python
gaussian_model = model_gaussian_rw(model_name='gaussian_w_uniform_prior',datafile_name=folder+filee)
gaussian_model.specify_model_subclass(0)
gaussian_model.load_data()
gaussian_model.initialize_parameter_range()
gaussian_model.initialize_transition()
gaussian_model.initialize_priors()
gaussian_model.run_inference()
gaussian_model.calc_ev()
gaussian_model.param_temp=10
gaussian_model.calc_prob_choice_softmax()
```

    180



```python
sns.set(rc={'image.cmap': 'cubehelix'})
sns.set_context('talk')
fig,ax = plt.subplots(1,1,figsize=(12,8))
ylims = ax.get_ylim()
xlims = [0,180]

# plot outcomes and true q
plt.plot(switch_model.data['probabilities'],color='r',label='true $q_t$')
x=switch_model.data['outcome']
plt.scatter(np.arange(len(x)),x,color='b',label='outcome $x_t\in{0,1}$')

# plot posterior
ax.imshow(gaussian_model.marg_r[:,0:-1],aspect='auto',extent=[xlims[0],xlims[1],ylims[0],ylims[1]],origin='lower')
ax.set_title('Posterior Distribution on Probability of Reward')
ax.set_xlabel('Trial')
ax.set_ylabel('Probability of Reward')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
```




    <matplotlib.legend.Legend at 0x1a68d30128>




![png](../../assets/images/Hierarchical_Bayesian_Filter_Models_files/Hierarchical_Bayesian_Filter_Models_26_1.png)


This model has the opposite problem as the 'switching model' in that it tends to underestimate $q$. This is because we are uncertain both about $q_t$ and about how much it is drifting $v_t$. This added uncertainty causes us to estimate that $q_t$ is closer to the center of its range (ie. 0.5).
