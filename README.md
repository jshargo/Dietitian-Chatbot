# Retrieval Augmented Dietitian Chatbot 

This project, which is in progress, aims to create an emotionally conscious dietitian chatbot support system to help people with lifestyle diseases. 

![Architecture](./Phase2.drawio.pdf)

## 🚀 Usage

Create a virtual environment and install the dependencies: (To be dockerized soon...)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
To run the backend:
```bash
uvicorn main:app --reload 
```

For the frontend:
```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To run the frontend:
```bash
flask run --debug 
```
Hosted on localhost:5000

To index your PDF documents:

For CSV file:
```bash
cd data 
python embed.py
```

To start Ollama server:
```bash
ollama pull gemma3
ollama serve
```
Restart ollama if port is already in use.


## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.



 
