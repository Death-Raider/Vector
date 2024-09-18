from transformer import Model
import numpy as np

def convert_to_str(data: dict)->str: # doesnt modify data
    out_str = ""
    for key in data.keys():
        out_str += key + ":" + str(data[key]) + ","
    out_str += " | "
    return out_str

def vectorize_database(model:Model, dataset: list[dict])->np.ndarray: # creates dataset
    embd_dataset = [0]*len(dataset)
    for index,data in enumerate(dataset):
        prompt = convert_to_str(data)
        embd = model.getEmbeddings(prompt)
        embd_dataset[index] = embd
    return np.array(embd_dataset)

def vectorize_database_search(model: Model, query, vector_db: np.ndarray):
    if type(query) == type([]):
        q_len = len(query)
        if type(query[0]) == type({}):
            query = list(map(convert_to_str, query))
        elif type(query[0]) == type(""):
            pass
        else:
            raise AttributeError("Query type not supported")
        
    elif type(query) == type({}):
        query = [convert_to_str(query)]

    elif type(query) == type(""):
        query = [query]

    else:
        raise AttributeError("Query type not supported")

    embd = model.getEmbeddings(query)
    same = model.compare(embd,vector_db).numpy()
    index = np.argmax(same,axis=1)
    print("Similarity for all data:", same)
    print("Similarity for all data:", index)
    # print("Max similarity: ",  same[0][index])
    return index

def main():
    model = Model()
    model.listModels()
    model.loadModel("paraphrase-MiniLM-L3-v2")
    database = np.array([
        {
            "product name": "Smartphone X",
            "product description": "A high-end smartphone with a powerful processor and stunning display.",
            "product category": "Electronics",
            "product cost": 799.99,
            "product id": "P1234"
        },
        {
            "product name": "Wireless Earbuds",
            "product description": "Bluetooth earbuds with noise cancellation and long battery life.",
            "product category": "Electronics",
            "product cost": 149.99,
            "product id": "P5678"
        },
        {
            "product name": "Laptop Pro",
            "product description": "A powerful laptop for work and gaming, featuring a high-resolution screen and fast SSD storage.",
            "product category": "Electronics",
            "product cost": 1299.99,
            "product id": "P9012"
        },
        {
            "product name": "Smartwatch Classic",
            "product description": "A classic smartwatch with fitness tracking and notification features.",
            "product category": "Wearables",
            "product cost": 249.99,
            "product id": "P3456"
        },
        {
            "product name": "Gaming Console",
            "product description": "A next-generation gaming console with high-definition graphics and exclusive games.",
            "product category": "Electronics",
            "product cost": 499.99,
            "product id": "P7890"
        }
    ])

    vector_db = vectorize_database(model, database)

    test_data = "Noise Cancelling and Good Sound Quality"

    index = vectorize_database_search(model,test_data,vector_db)
    print("Query: ", test_data)
    print("Likey Product: ", database[index])

if '__main__' == __name__:
    main()