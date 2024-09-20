# Prompt Embedding API for Semantic Search in Vector Databases
Hosts Hugging-Face sentence-transformers, which are specifically trained for embedding creation, on local host (127.0.0.1) at port 3000.
## Install requirements
```shell
pip install -r requirements.txt
```
## Run API
```shell
python3 API.py
```
## Fetch Request
**Destination**: localhost:3000/vectorEmbeddings <br>
**Method**: POST <br>
**Body**: JSON object <br>
```js
fetch('http://localhost:3000/vectorEmbeddings', {
  method: 'POST',
  mode: 'no-cors',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    product_name: "Smartphone X",
    product_description: "A high-end smartphone with a powerful processor",
    product_category: "Electronics",
    product_cost: 799.99,
    product_id: "P1234"
  })
})
.then(response => {
  if (!response.ok) {
    throw new Error('Network response was not ok');
  }
  return response.json();
})
.then(data => { console.log(data) })
.catch(error => {  console.error('Error:', error) });
```
## Return Data
Returns a JSON object with the same details as the POST body with two new keys: <br>
<ul>
    <li>vector: Embedding Vector</li>
    <li>inferenceTime: Time taken for the model to generate the embedding vector</li>
</ul>