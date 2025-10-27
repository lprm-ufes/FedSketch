## FedSketch: Federated Learning with Privacy and Communication Efficiency

This repository contains the implementation of **FedSketch**, an innovative algorithm for federated learning that aims to **enhance data privacy and communication efficiency**. FedSketch uses **probabilistic data structures called “sketches”** to compress the weight vectors of machine learning models, significantly reducing the amount of data transmitted between the server and clients during the training process.

### Main Features

* **Model Compression:** FedSketch compresses machine learning models using sketches, **reducing the transmitted data size by up to 73×.**
* **Differential Privacy:** The probabilistic nature of sketches ensures a high level of differential privacy, with ϵ values reaching 10⁻⁶, protecting client data against inference attacks.
* **Communication Efficiency:** Model compression results in more efficient communication between the server and clients, making FedSketch ideal for bandwidth-constrained environments.
* **Accuracy Preservation:** Despite compression, FedSketch maintains accuracy comparable to traditional federated learning, particularly on datasets such as MNIST.

### Architecture

FedSketch follows the standard client-server architecture of federated learning, with the following main steps:

1. **Local Training:** Each device in the federation trains a neural network using only its local data.  
2. **Compression and Transmission:** The locally trained neural network parameters are compressed into a sketch and transmitted to the aggregation server.  
3. **Aggregation:** The aggregation server combines the sketches received from the clients to create a global model.  
4. **Decompression and Update:** The aggregation server shares the global model with the devices, which decompress the sketch and retrain their local models with new data.  

### Benefits

* **Reduced Communication Costs:** Model compression greatly reduces communication overhead, making federated learning more feasible in real-world scenarios.  
* **Enhanced Privacy:** The inherent differential privacy of sketches ensures strong data protection, making FedSketch suitable for privacy-sensitive applications.  
* **Scalability:** FedSketch can easily scale to a large number of clients, making it suitable for large-scale federated learning applications.  

### Applications

FedSketch can be applied across multiple domains where data privacy and communication efficiency are critical, including:

* **Healthcare:** Training machine learning models on distributed electronic health records without compromising patient privacy.  
* **Finance:** Fraud detection and other machine learning models on distributed financial data.  
* **Internet of Things:** Training machine learning models on IoT devices with limited computational resources.  

### Limitations

* FedSketch may take longer to converge compared to traditional federated learning on certain datasets.  
* The optimal choice of sketch parameters (number of hash maps and table sizes) depends on the dataset and requires experimentation.  

### Future Work

* Investigate methods to automatically optimize sketch parameters based on dataset characteristics.  
* Explore the integration of FedSketch with other privacy-preserving techniques, such as homomorphic encryption.  
* Evaluate FedSketch performance in more complex scenarios, such as federated learning with non-IID data.  

### Conclusion

FedSketch is a promising approach to federated learning that addresses key concerns of privacy and communication efficiency. By compressing models using sketches, FedSketch significantly reduces the amount of transmitted data while preserving model accuracy and protecting data privacy. The potential applications of FedSketch are vast, spanning multiple domains where privacy and efficiency are critical.
