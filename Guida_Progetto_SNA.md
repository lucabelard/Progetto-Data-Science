# Guida al Progetto di Social Network Analysis (SNA) e Valutazione del Dataset

Questa guida sintetizza:
1. **I contenuti teorici della Social Network Analysis** trattati nelle slide del corso (*Parte II: Social Network Analysis*, Prof. Domenico Ursino, A.A. 2026/2027).
2. **In cosa consiste concretamente il progetto**, analizzando i due elaborati d'esempio dello stesso corso UNIVPM (*Calciomercato Transfermarkt* e *Harry Potter*).
3. **Una valutazione tecnica e onesta** del dataset proposto in `datasetexample.md` (*Operazione Montagna*).

---

## 1. Cos'è la Social Network Analysis (SNA)

La **Social Network Analysis (SNA)** è una metodologia interdisciplinare (nata nella sociologia ben prima di Facebook e X/Twitter) definibile come lo **studio delle relazioni umane — o tra entità — per mezzo della teoria dei grafi**. L'assunto centrale è che *le relazioni, prese nel loro insieme, definiscono chi sono gli attori, qual è il loro potere e come agiscono*.

### 1.1 Concetti fondativi (dalle slide del corso)
- **Diade, Nodi e Archi**: l'unità minima è la *diade* (es. `Alice – comunica con – Bob`). Gli attori sono i **nodi** (o vertici) e le relazioni sono gli **archi** (o link).
- **Tipi di relazione**:
  - **Binarie** (esiste/non esiste) vs **Pesate** (hanno un'intensità, spesso misurata tramite frequenza di interazione — es. *scala oggettiva di Krackhardt*, preferita alla scala di Likert).
  - **Simmetriche** (grafo *non diretto*, es. amicizia su Facebook) vs **Asimmetriche** (grafo *diretto*, es. follow su X, trasferimenti tra club, telefonate chiamante $\to$ chiamato).
  - **1-mode** (un solo tipo di nodo) vs **2-mode / bipartiti** (due tipi di nodi, es. persone e club) vs **Multimodali** (più tipi di nodi/relazioni).
- **Reti informali vs strutture formali**: le reti reali non coincidono quasi mai con l'organigramma ufficiale (caso *ACME Consulting / Frida*: il vero perno aziendale era una segretaria storica, non il CEO). Nascono spontaneamente anche in ambienti ostili (es. prigione *Butyrka*) e mostrano **resilienza** agli attacchi (es. cellule di *Al Qaeda*: rimuovere il leader/broker scollega temporaneamente la cellula, ma legami informali la riconnettono).
- **Weak Ties (Granovetter) e Numero di Dunbar**:
  - I **legami deboli (*weak ties*)** hanno basso costo emotivo ma trasportano informazioni **nuove** tra comunità distanti (es. trovare lavoro o amplificare mobilitazioni come le rivoluzioni di Twitter / Piazza Tahrir, guidate dalla *"middle class"* immersa in cluster densi più che dalle celebrità).
  - Il **Numero di Dunbar ($\approx 150$)** rappresenta il limite cognitivo di relazioni stabili umane, organizzato a piramide: cerchia intima ($\approx 7$), famiglia estesa, tribù e infine *weak ties*.

### 1.2 Teoria dei Grafi e Metriche Globali
- **Rappresentazione**: *Matrice di adiacenza* (inefficiente su reti reali sparse, spesso con densità $\sim 0.1\%$) vs *Liste di adiacenza / edgelist*.
- **Indici di base**:
  - **Size**: numero di nodi $|V|$ e di archi $|E|$.
  - **Degree** (grado): numero di archi incidenti (in grafi diretti: **in-degree** e **out-degree**); **mean degree**.
  - **Densità**: rapporto tra archi esistenti e archi possibili (esiste anche la versione pesata).
- **Attraversamento e Distanze**:
  - **DFS** (*Depth-First Search*) e **BFS** (*Breadth-First Search*).
  - **Walk, Path, Ciclo, Trail, Tour**.
  - **Distanza geodesica**: lunghezza dello *shortest path* tra due nodi (calcolabile con **Dijkstra** sui grafi pesati).
  - **Eccentricità** di un nodo (massima distanza geodesica dagli altri nodi), **Raggio** (eccentricità minima), **Diametro** (eccentricità massima) e **Periferia** (nodi con eccentricità pari al diametro). *Nota: raggio e diametro sono finiti solo su grafi connessi (o sulla Giant Connected Component).*

### 1.3 Proprietà macroscopiche delle reti sociali reali
1. **Six Degrees of Separation & Small-World Networks**: esperimento di Milgram (catena media $5.5$); le reti sociali reali sono *small-world*: combinano alto clustering locale (comunità dense) con pochi archi "a lungo raggio" (*boundary spanners*) che abbattono il diametro.
2. **Paradosso dell'amicizia**: in media i tuoi amici hanno più amici di te ($\mathbb{E}[f(Y)] \ge \mathbb{E}[f(X)]$), perché campionando un arco si intercettano più facilmente gli hub ad alto grado.
3. **Distribuzione Power-Law (Scale-Free / Pareto)**:
   - A differenza della distribuzione normale, il grado nelle reti reali ha una **coda pesante (*heavy-tailed*)**: $f_X(x) = C x^{-(\alpha+1)}$.
   - È **scale-free** (autosimile a tutte le scale) e segue spesso la regola empirica 80/20.
   - Si verifica con **grafico log-log** (retta con pendenza $-(\alpha+1)$), usando preferibilmente la **funzione di sopravvivenza (CCDF)** per ridurre il rumore nella coda, oppure con un **grafico di Pareto**.
   - Nasce dal meccanismo di crescita + **attaccamento preferenziale (*preferential attachment*)**: *"i ricchi diventano più ricchi"* (con *vantaggio del primo arrivato*).
4. **Campionamento**: necessario quando la rete originale è troppo grande per il calcolo esatto delle metriche.

### 1.4 Misure di Centralità (chi conta davvero nella rete?)
Ogni misura cattura un diverso tipo di "potere" strutturale:

| Metrica | Metafora del corso | Cosa misura |
| :--- | :--- | :--- |
| **Degree Centrality** | **Celebrità / Hub** | Numero di connessioni dirette. Identifica i nodi più popolari/attivi, ma ignora la posizione globale. |
| **Closeness Centrality** | **Mercanti di gossip** | Reciproco della somma delle distanze minime dagli altri nodi. Misura quanto velocemente un nodo raggiunge l'intera rete. Distribuzione più vicina alla normale. |
| **Betweenness Centrality** | **Colli di bottiglia / Bridge** | Frazione di *shortest path* tra tutte le coppie di nodi che passano per il nodo. Identifica mediatori, broker e punti vulnerabili che collegano comunità diverse. |
| **Eigenvector Centrality** *(e PageRank)* | **Eminenze grigie** *(es. Don Corleone)* | Un nodo è importante se è connesso ad altri nodi importanti. Cattura il potere indiretto "dietro le quinte". |
| **Geometric Potential Gain (GPG)** | **Influenza potenziale con decadimento** | *(Novità nelle slide 2026/2027)* Somma dei nodi/cammini a distanza $k$ pesati con decadimento geometrico $\delta^{k-1}$ ($0 < \delta < 1$). Unisce accessibilità locale ed estesa attenuando l'effetto dei nodi lontani. |

### 1.5 Strutture Mesoscopiche e Locali
- **Componenti connesse**: sottografi massimali connessi (in grafi diretti: *componenti deboli* vs *forti*).
- **Ego Network**: sottorete centrata su un nodo (*ego*) e sui suoi vicini (*alters*) fino a raggio $\le 3$ (legato all'*orizzonte di osservabilità*), includendo gli archi tra gli alters.
- **Clustering Coefficient**: frazione di vicini di un nodo connessi anche tra loro (alto = forte fiducia reciproca/coesione; basso = struttura a stella / broadcaster).
- **Triadi, Buchi Strutturali e Triad Census**:
  - Insiemi di 3 nodi; le triadi chiuse sono le unità più stabili.
  - Le triadi aperte formano **buchi strutturali (*structural holes*)** sfruttati dai mediatori.
  - Su grafi diretti esistono **16 configurazioni triadiche** (*Triad Census*, codifica M-A-N + U/D/C/T).
  - **Equilibrio strutturale e conflitti**: *"l'amico del mio amico è mio amico"* (chiusura triadica $\to$ crescita virale superata la soglia di densità del **7–10%**) e *"il nemico del mio amico è mio nemico"* (propagazione dei conflitti $\to$ *criticità auto-organizzata*).
- **Clique e rilassamenti**:
  - **Clique**: sottografo completo massimale (tutti connessi a tutti).
  - Rilassamenti: **N-clique** (distanza $\le N$), **N-clan** (percorsi tutti interni al sottografo), **k-plex** (connesso ad almeno $|S|-k$ membri), **k-core** (ogni nodo ha grado interno $\ge k$; il massimo $k$ è il **Main Core**).
- **Community Detection**: partizionamento della rete in gruppi densamente connessi internamente e debolmente verso l'esterno (es. **Louvain**, che massimizza la *modularità*, e **Girvan-Newman**, che rimuove iterativamente gli archi con massima *edge betweenness*).

### 1.6 Diffusione delle Informazioni, Modelli Epidemici e Reti 2-Mode
- **Information Diffusion**: passaggio da crescita lineare (processo di Poisson nella propria ego-network) a virale una volta superata la **massa critica** ($\sim 6\text{--}7\%$, es. lancio di Facebook a Harvard) fino alla **saturazione**; ruolo dei costi di partecipazione e della qualità del contenuto ($V_m$).
- **Modelli Compartimentali Epidemici**:
  - **SIS** ($S \to I \to S$, senza immunità) e **SIR** ($S \to I \to R$, con immunità permanente; estendibile a SEIR).
  - Parametri chiave: tasso di trasmissione $\beta$, tasso di guarigione $\gamma = 1/D$, numero riproduttivo di base $\mathcal{R}_0 = \beta/\gamma$, soglia $N/\mathcal{R}_0$, severità dell'epidemia.
  - Su rete sociale il contagio avviene solo lungo gli archi $\to$ emergono i **superdiffusori** (spesso regola 80/20).
- **Omofilia vs Curiosità**: *omofilia di stato* vs *di valore*, bilanciata dalla *curiosità* e limitata dalla *trappola della noia*; ruolo dei *boundary spanners*.
- **Two-Mode (Bipartite) e Multimodal Networks**: reti di affiliazione (es. Persone–Club $PC$), proiettabili su reti 1-mode via moltiplicazione matriciale ($PP = PC \cdot PC^T$ e $CC = PC^T \cdot PC$) per inferire legami da co-appartenenza.

---

## 2. In cosa consiste il Progetto (Analisi dei 2 PDF d'esempio)

Entrambe le relazioni di esempio provengono dallo **stesso corso e dagli stessi docenti** (UNIVPM – Prof. Domenico Ursino e Dott. Michele Marchetti, A.A. 2024/2025) e usano **Python + `NetworkX`** (+ `pandas`, `matplotlib`, `seaborn`, `python-louvain`).

### 2.1 Confronto tra i 2 progetti d'esempio

| Aspetto | Esempio 1: Calciomercato Transfermarkt (`Social-Network-Analysis.pdf`) | Esempio 2: Saga "Harry Potter" (`Social_network_analysis.pdf`) |
| :--- | :--- | :--- |
| **Lunghezza** | **145 pagine** *(molto corposo, anche perché ripete i grafici su 9 sottografi)* | **36 pagine** *(snello ed essenziale)* |
| **Dataset** | Kaggle *Transfermarkt* (10 CSV: club, trasferimenti, valori economici, campionati) | Dataset personaggi di Harry Potter (2 CSV: `characters.csv`, `relations.csv`) |
| **Scala del grafo** | **439 nodi, 13.183 archi** (più sottografi nazionali da ~33–39 nodi e `G_expensive` da 36 nodi) | **65 nodi, 330 archi** |
| **Tipo di grafo** | **Diretto** (`DiGraph`), **pesato** (numero trasferimenti + costo economico) | **Non diretto** (`Graph`), **pesato** (pesi 0–4 derivati in preprocessing da `bio` + polarità `+`/`-`) |
| **Metriche globali** | Nodi, archi, densità, clustering medio, raggio, diametro (su grafo globale e 8 varianti/sottografi) | Nodi, archi, densità semplice e pesata, diametro, raggio, periferia, average clustering, connettività |
| **Centralità** | Degree, Betweenness, Eigenvector, **PageRank** *(Closeness omessa perché simile a Betweenness)* | Degree, **Closeness**, Betweenness, Eigenvector |
| **Strutture analizzate** | **Community (Louvain)** + validazione su campionati, bilanci economici e cosine similarity; **Cliques** + intersezioni/hub; **K-core (Main Core)**; **Ego-network** (scelta tramite media delle centralità) | **Triadi**; **Cliques** (totale, $\ge 4$, clique massima di 11 nodi); **K-core (Main Core)**; **Ego-network** (Bellatrix Lestrange); **Community** (confronto **Girvan-Newman vs Louvain** sul sottografo studenti rispetto alle 4 Casate) |
| **Analisi avanzate** | **Link Prediction** con **Node2Vec** (embedding 64D, visualizzazione t-SNE, cosine similarity, predizione nuovi archi) | Nessuna estensione predittiva o simulativa |

### 2.2 Struttura attesa del nostro progetto
Dalla lettura incrociata delle slide e dei 2 PDF emerge una **struttura standard molto chiara** in 5 blocchi:

1. **Introduzione, Dataset & Modellazione del Grafo**
   - Motivazione della scelta del dataset, descrizione di tabelle/attributi, operazioni di **preprocessing** e pulizia.
   - Definizione rigorosa di **nodi, archi, eventuale direzione e pesi**, oltre alla costruzione di eventuali **sottografi o layer** tematici significativi.
2. **Analisi Esplorativa e Proprietà Globali della Rete**
   - Calcolo e commento di: numero di nodi/archi, grado medio, densità (semplice/pesata), componenti connesse, diametro, raggio, periferia, average clustering coefficient.
   - Analisi della **distribuzione dei gradi** (verifica empirica *Power-Law* / *log-log plot* / *Pareto*, se la scala lo permette) e visualizzazioni ragionate (es. *spring*, *Kamada-Kawai*, *circular*).
3. **Analisi delle Misure di Centralità**
   - Calcolo di **Degree, Closeness, Betweenness, Eigenvector** (più eventuale **PageRank** e il **Geometric Potential Gain – GPG**, presente nelle slide 2026/2027 e quindi un ottimo elemento distintivo!).
   - Per ciascuna metrica: Top/Bottom nodi, grafico di distribuzione, grafo colorato per valore e soprattutto **interpretazione di dominio** (chi sono le celebrità, i bridge, le eminenze grigie e perché le classifiche differiscono o coincidono).
4. **Analisi Strutturale (Micro e Mesoscopica)**
   - **Triadi** (e *Triad Census* se il grafo è diretto) e coefficienti di clustering.
   - **Cliques** (distribuzione per taglia, clique massima, nodi ponte tra più clique).
   - **K-core / Main Core** (isolamento del nucleo più denso).
   - **Ego Network** di attori particolarmente significativi a confronto.
   - **Community Detection** (confronto **Louvain vs Girvan-Newman** con modularità) e **validazione esterna**: verificare se le comunità scoperte dalla topologia corrispondono ad attributi reali del dataset (come campionati/bilanci nell'Esempio 1 o le Casate nell'Esempio 2).
5. **Estensione Avanzata / Valore Aggiunto** *(ciò che fa la differenza tra un progetto "da 25" e un progetto "da 30 e lode")*
   - A seconda del dataset: **Network Disruption / analisi di resilienza** (rimozione mirata dei nodi), **simulazione di diffusione (SIR/SIS)** sul grafo reale, **Link Prediction** (euristiche topologiche o Node2Vec), oppure **analisi 2-mode / proiezione matriciale**.

---
