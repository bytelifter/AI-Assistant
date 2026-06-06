# Engine SpA AI Assistant

Architettura RAG modulare per interrogare rapidamente i documenti aziendali presenti in `KNOLEDGE/`.

## Struttura

- `main.py` - entry point Streamlit
- `services.py` - logica di retrieval e chiamata al modello
- `core/config.py` - configurazione centralizzata da variabili d'ambiente
- `core/vector_store.py` - build/load del database vettoriale Chroma
- `core/exceptions.py` - eccezioni applicative
- `utils/response_parser.py` - normalizzazione e parsing risposte
- `ingest.py` - costruzione dell'indice vettoriale
- `db_engine/` - database vettoriale persistente

## Flusso

1. `ingest.py` legge i file in `KNOLEDGE/`
2. I documenti vengono spezzati in chunk e salvati in `db_engine/`
3. `main.py` carica il DB vettoriale persistente
4. `services.py` esegue similarity search e interroga OpenRouter
5. `utils/response_parser.py` normalizza la risposta e raccoglie le fonti

### Embeddings

Il progetto usa embeddings locali basati su `sentence-transformers`.

## Avvio

1. Copiare `.env.example` in `.env`
2. Compilare la chiave OpenRouter
3. Installare le dipendenze
4. Eseguire:
   - `python ingest.py`
   - `streamlit run main.py`

## Note

- L'indice vettoriale evita di rileggere tutti i file a ogni domanda.
- La cartella `KNOLEDGE/` è trattata come sorgente documentale del bot.
- `db_engine/` deve esistere e contiene l'indice persistente Chroma.
- Per estendere il progetto, aggiungere nuovi loader in `core/vector_store.py` o nuovi servizi in `services.py`.
