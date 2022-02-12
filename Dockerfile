FROM 3.11-rc-bullseye
COPY api_app/main.py main.py
COPY api_app/exported_classifier.pickle exported_classifier.pickle
COPY api_app/exported_one_hot.pickle exported_one_hot.pickle
COPY requirements.txt requirements.txt

#COPY . .

RUN pip install -r requirements.txt

ENV FLASK_APP=main
CMD ["flask", "run"]
