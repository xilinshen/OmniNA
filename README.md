<!DOCTYPE html> <html lang="en"> 
<head>     
<meta charset="UTF-8">     
<meta name="viewport" content="width=device-width, initial-scale=1.0">     
</head> 
<body>     
<div align="center">         
<h1> 🌐 OmniNA</h1>         
<p>A Foundation Model for Nucleotide Sequences</p>     
</div>      
<h2>📋 Overview</h2>     
<p> OmniNA is a foundation generative model designed for comprehensive nucleotide sequence learning. OmniNA is pre-trained on all nucleotide sequences and the corresponding annotations obtained from NCBI nucleotide database. The model can be fine-tuned for various genomic tasks.</p>
    
<h2>✨ Features</h2>
<ul>
<li>Pre-trained on 91.7 million nucleotide sequences from various species.</li>
<li>Fine-tuned across 17 tasks, achieving state-of-the-art performance.</li>         
<li>Designed to be scalable and accessible to users of all expertise levels.</li>     
</ul>

<h2>🔗 Model Variants</h2>
<p>OmniNA comes in three variants with different parameter scales to meet various needs:</p>
<table>
<thead>
<tr>
<th>Model Variant</th><th>Number of Layers</th><th>Number of Heads</th><th>Number of Parameters</th>
</tr></thead>
<tbody>
<tr><td>OmniNA-66M</td><td>8</td><td>8</td><td>66 Million</td></tr>
<tr><td>OmniNA-330M</td><td>16</td><td>16</td><td>330 Million</td></tr>
<tr><td>OmniNA-1.7B</td><td>24</td><td>32</td><td>1.7 Billion</td></tr>
</tbody>
</table>      

<h2>🔨 Installation</h2>
<p>Required packages can be installed using:</p>
<pre><code>pip install datasets
pip install torch==2.3.1
pip install transformers==4.41.2
pip install accelerate==0.31.0
pip install tensorboardX==2.6.2.2</code></pre>
<p>To utilize the pre-trained OmniNA model, begin by cloning the Hugging Face repository:</p>
<pre><code>XLS/OmniNA-1.7B
XLS/OmniNA-220m
XLS/OmniNA-66m 
</code></pre>      
<p></p>
<h2>🌸 Training</h2>
<h3>Prepare Data</h3>
<p>Following these steps to prepare your data:</p>
<ol>
<li><strong>Collecting Data:</strong> Map nucleotide sequences to their corresponding annotations.</li>
<li><strong>Chunk Sequences:</strong> Ensure the sequences are between 200 bp and 3000 bp in length.</li>
<li><strong>Formatting Data:</strong> You should prepare the data as a .csv file for both pretraining and fine-tuning tasks. Here's how you can organize the data:</li>
<ul>
<li><strong>For pretraining (two columns):</strong></li>
<ul>
    <li><strong>sequence:</strong> The nucleotide sequence.</li>
    <li><strong>response:</strong> The corresponding sequence annotation.</li>
</ul>
<p>For example:</p>
<table>
<thead>
<tr>
<th>sequence</th><th>response</th>
</tr>
</thead>
<tbody>
<tr>
<td>ATGCATGCATGC</td><td>Annotation A</td>
</tr>
<tr>
<td>GCTAGCTAGCTA</td><td>Annotation B</td>
</tr>
<tr>
<td>TAGCTAGCTAGC</td><td>Annotation C</td>
</tr>
</tbody>
</table>
<li><strong>For pretraining (three columns):</strong></li>
<ul>
<li><strong>sequence:</strong> The nucleotide sequence.</li>
<li><strong>question:</strong> The question related to the sequence.</li>
<li><strong>response:</strong> The corresponding answer.</li>
</ul>
<p>For example:</p>
<table>
<thead>
<tr>
<th>sequence</th><th>question</th><th>response</th>
</tr>
</thead>
<tbody>
<tr>
<td>ATGCATGCATGC</td><td>Is the sequence identified as a promoter?</td><td>Yes</td>
</tr>
<tr>
<td>GCTAGCTAGCTA</td><td>Is the sequence identified as an enhancer?</td><td>No</td>
</tr>
</tbody>
</table>
</ul>
</ol>

<h3>Pretrain</h3>
<p>
<code>bash pretrain.sh</code>
</p>
<p>Output of first stage of training will be saved in `output/pretrain/`.</p>

<h3>Finetune</h3>
<p>
<code>bash finetune.sh</code>
</p>
<p>The final trained model checkpoint will be saved in `output/finetune`.</p>

<h2>📚 Citation</h2>     
<p>If you use OmniNA in your research, please cite our work:</p>     
<pre><code>@article{Shen2024.01.14.575543,
title={OmniNA: A Foundation Model for Nucleotide Sequences},
author={Xilin Shen, Xiangchun Li},
journal={bioRXiv},
year = {2024},
doi = {10.1101/2024.01.14.575543}}</code></pre>      

<h2>🌟 Acknowledgement</h2>     
<p>The codebase we built upon is adapted from <a href="https://github.com/meta-llama/llama">Llama</a>. We thank the original authors for their valuable contributions!</p>     
<h2>📄 License</h2>     
<p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p> 
</body> 
</html>


