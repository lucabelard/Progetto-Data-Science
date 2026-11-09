# Guida Completa e Piano Operativo – Parte 2: Analisi delle Serie Storiche sul Mercato Calcistico (Transfermarkt)

Questo documento raccoglie in modo **chiaro, semplice, discorsivo ed elementare** tutte le scelte progettuali, le verifiche fatte sui dati reali e il piano di lavoro per la **Parte 2 del progetto di Data Science** (dedicata alle *Time Series*).

L'obiettivo di questa guida è duplice:
1. Fissare passo dopo passo **cosa andremo a fare sul dataset di Transfermarkt**, dando massimo spazio alle parti più belle e centrali del lavoro: **ETL (Data Engineering)**, **Analisi Esplorativa (EDA)** e **Analisi degli Shock Storici**.
2. Spiegare **il perché di ogni scelta** con esempi pratici e numeri reali estratti dal dataset, così che ogni componente del gruppo possa padroneggiare il lavoro e spiegarlo a voce al Prof. / Dott. Christopher Buratti durante l'esame orale.

---

## 1. Il Contesto: I Paletti del Professore e Perché Abbiamo Scelto "Valore delle Rose e Spese di Mercato"

Nello scambio di mail con il Dott. Buratti sono emerse **tre indicazioni fondamentali** che guidano tutto il nostro lavoro:
1. **Scegliere UNA SOLA direzione e approfondirla bene** (*"Di tutte queste alternative dovete sceglierne una e proseguire in quella direzione"*).
2. **Garantire intervalli temporali regolari** (*"Andrebbe verificato innanzitutto che i dati temporali che si hanno sono a intervalli regolari, altrimenti non riusciamo a costruire una serie temporale"*).
3. **Garantire che la serie non sia piatta/costante nel tempo** (*"E che i valori della serie non risultino essere quasi costanti nel tempo, altrimenti la serie sarebbe praticamente piatta"*).

### Perché abbiamo scartato definitivamente Gol, Disciplina e Affluenza negli Stadi
Dopo aver scaricato ed esplorato i 12 file CSV reali del dataset di Transfermarkt (`davidcariboo/player-scores`), abbiamo verificato con i numeri alla mano perché le altre piste erano rischiose:
* **Media Gol e Cartellini (Scartati):** In 14 anni la media gol nei campionati europei oscilla quasi sempre tra 2,6 e 2,8 gol a partita. Vista su un grafico temporale è praticamente una linea piatta con piccole oscillazioni casuali (rumore bianco). Un modello predittivo su una serie del genere tirerebbe fuori una linea orizzontale inutile.
* **Affluenza negli Stadi (Scartata definitivamente):** Guardando i dati veri di `games.csv`, l'affluenza ha **tre gravi difetti strutturali** per una serie storica classica:
  1. **Il buco estivo:** a giugno e luglio non si gioca quasi mai, quindi mancano 2 mesi su 12 ogni anno (gli intervalli mensili non sono regolari).
  2. **Le anomalie di calendario:** in certi mesi di confine si gioca una sola partita di recupero (es. a **Giugno 2024** in Serie A si è giocata *solo* Atalanta-Fiorentina a Bergamo davanti a **14.851 spettatori**, facendo sembrare che la media della Serie A sia crollata da 31.668 a 14.851 spettatori solo perché non hanno giocato San Siro o l'Olimpico!).
  3. **L'azzeramento totale per il COVID:** da marzo 2020 a metà 2021 ci sono **14 mesi consecutivi con 0 spettatori**. Un buco piatto a zero lungo più di un anno nel mezzo della serie manda in crisi i modelli matematici classici (ARIMA/SARIMA).

### Perché il focus "Valore di Mercato delle Rose + Volume Spese" è la scelta perfetta
Trattare il **Valore di Mercato delle Rose (`player_valuations.csv`)** affiancato dal **Volume delle Spese di Calciomercato (`transfers.csv`)** risolve tutti i problemi:
1. **Intervalli regolari veri (12 mesi su 12):** Una squadra di calcio e i suoi giocatori hanno un valore economico **anche a giugno e luglio**! Non ci sono buchi estivi: da **Gennaio 2012 a Giugno 2026** abbiamo **174 mesi consecutivi** perfettamente equispaziati ($\Delta t = 1\text{ mese}$).
2. **I dati sono abbondantissimi (confronto con i progetti d'esempio del corso):**
   * Nei progetti d'esempio del corso (es. *Apple* e *S&P 500 / Nvidia*), i dati giornalieri di borsa venivano aggregati a livello mensile o bisettimanale su 10–14 anni, ottenendo circa **120–168 punti**.
   * Noi partiamo da **656.301 valutazioni di mercato**, **oltre 80.000 trasferimenti** e **1,6 milioni di presenze in partita**, che aggregati mese per mese dal 2012 al 2026 ci danno **174 punti mensili consecutivi** (oltre 14 cicli annuali completi da 12 mesi!).
3. **Analogia perfetta con la Finanza:** È come analizzare l'indice di borsa **S&P 500 del calcio europeo**:
   * Il **Valore delle Rose ($Y$)** è il valore dell'indice azionario (quanto valgono tutti gli asset dei club in quel mese).
   * Il **Volume delle Spese di Trasferimento ($X$)** è il `Volume` degli scambi di borsa (quanti milioni di euro vengono movimentati sul mercato in quel mese).

---

## 2. FASE 1 – ETL e Data Engineering: Come Costruiamo la Serie Storica (Spiegato Passo per Passo)

Questa è una delle parti più importanti del progetto, perché trasforma un database di tabelle separate in una **serie storica mensile pulita, continua e mai piatta**. Inoltre, mostra al professore che abbiamo capito a fondo come sono fatti i dati di Transfermarkt senza cadere nelle trappole più comuni.

### 2.1 Le Tabelle di Partenza che Utilizziamo
Dal dataset grezzo (`data/football/raw/`) usiamo 5 tabelle:
1. **`player_valuations.csv` (656.301 righe):** contiene lo storico delle valutazioni (`player_id`, `date`, `market_value_in_eur`). Ogni volta che Transfermarkt aggiorna il prezzo di un calciatore, viene aggiunta una riga con la data e il nuovo valore in euro.
2. **`transfers.csv`:** contiene tutti i trasferimenti dei calciatori da un club all'altro (`player_id`, `transfer_date`, `from_club_id`, `to_club_id`, `transfer_fee`, `market_value_in_eur`).
3. **`appearances.csv` e `games.csv`:** contengono tutte le partite giocate e i giocatori scesi in campo con la maglia di ciascun club in una determinata data.
4. **`clubs.csv` e `competitions.csv`:** contengono l'anagrafica dei club e il campionato nazionale di appartenenza:
   * **`IT1`**: Serie A (Italia)
   * **`GB1`**: Premier League (Inghilterra)
   * **`ES1`**: LaLiga (Spagna)
   * **`L1`**: Bundesliga (Germania)
   * **`FR1`**: Ligue 1 (Francia)

### 2.2 I Due Problemi dei Dati Grezzi (e Come li Risolviamo nell'ETL)

Se uno studente inesperto prendesse `player_valuations.csv` e facesse semplicemente `groupby('mese').sum()`, commetterebbe **due errori gravissimi**. Vediamoli con un esempio concreto:

* **Problema 1 – Gli aggiornamenti di prezzo non avvengono tutti i mesi per tutti i giocatori:**
  Se guardiamo un singolo calciatore (es. Lautaro Martínez), Transfermarkt non scrive una riga nuova nel CSV ogni giorno o ogni mese, ma aggiorna il suo cartellino **3 o 4 volte l'anno** (tipicamente a **Marzo**, **Maggio/Giugno**, **Ottobre** e **Dicembre**).
  * *Cosa succederebbe senza ETL?* A Dicembre (quando Transfermarkt aggiorna 3.000 giocatori) la somma farebbe 30 miliardi; a Gennaio (quando Transfermarkt aggiorna solo 6 giocatori) la somma crollerebbe a 20 milioni!
  * *Come lo risolviamo:* Se un giocatore viene valutato 50 milioni a Dicembre e la sua valutazione successiva è a Marzo (60 milioni), nei mesi di Gennaio e Febbraio quel giocatore **non vale zero**: vale ancora **50 milioni**! Quindi, per ogni singolo calciatore, costruiamo una griglia mensile continua in cui il suo ultimo valore di mercato noto viene mantenuto valido (*forward-fill*, con un limite massimo di sicurezza di 12 mesi per non trascinare giocatori ritirati) fino al successivo aggiornamento di prezzo.

* **Problema 2 – I giocatori cambiano squadra durante il calciomercato (Gennaio e Estate):**
  Nella tabella `player_valuations.csv`, la colonna `current_club_id` indica solo la squadra in cui il giocatore gioca **oggi (nel 2026)**, non la squadra in cui giocava nel 2015 o nel 2018! Inoltre, nei mesi di **Gennaio-Febbraio** e **Luglio-Agosto**, Transfermarkt fa pochissimi aggiornamenti di prezzo perché aspetta la chiusura del calciomercato.
  * *Come lo risolviamo:* Incrociamo la storia del giocatore con **`transfers.csv`** e **`appearances.csv`** per sapere **in quale club e in quale campionato militava realmente quel giocatore in quel preciso mese storico**!
  * **Il risultato magico di questo incrocio:**
    1. Nei mesi di campionato (**Marzo, Maggio/Giugno, Ottobre, Dicembre**), il valore della rosa cambia perché **Transfermarkt aggiorna il valore dei cartellini** (chi gioca bene sale di prezzo, chi invecchia o gioca male scende).
    2. Nei mesi di calciomercato (**Gennaio-Febbraio** e **Luglio-Agosto-Settembre**), il valore della rosa cambia perché **le squadre comprano e vendono migliaia di giocatori** (`transfers.csv` registra oltre **5.000 trasferimenti a luglio** e **3.500 a gennaio**!).
    3. Mettendo insieme i due effetti, la nostra serie storica mensile **si muove tutti i 12 mesi dell'anno**, senza mai restare piatta!

### 2.3 Cosa conterrà la Tabella Finale Pulita (`market_monthly.csv`)
Alla fine dell'ETL avremo una tabella pulitissima con **una riga per ogni mese** (da `2012-01` a `2026-06`, **174 righe totali**, zero valori mancanti) e queste colonne:
1. **`date` (`year_month`)**: il mese di riferimento (es. `2012-01`, `2012-02`, ..., `2026-06`).
2. **`top5_market_value_bn`**: Valore di mercato totale delle rose dei **Top-5 Campionati Europei** (in Miliardi di €).
3. **`serie_a_market_value_bn`** (e lo stesso per Premier League, LaLiga, Bundesliga, Ligue 1): Valore di mercato totale delle rose della **Serie A** (in Miliardi di €).
4. **`mean_player_value_m`**: Valore medio per calciatore attivo nei Top-5 campionati (in Milioni di €).
5. **`transfer_spend_m` (`Volume Spese`)**: Somma di tutti i costi di trasferimento (`transfer_fee > 0`) spesi nel mese (in Milioni di €).
6. **`n_transfers` (`Volume Operazioni`)**: Numero totale di trasferimenti completati nel mese.
7. **`pct_change_top5` e `pct_change_serie_a` (`Change %`)**: Variazione percentuale del valore delle rose rispetto al mese precedente.
8. **Variabili Dummy degli Shock Storici (`dummy_neymar_post2017`, `dummy_covid_2020`, `dummy_transfer_window`)**: indicatori `0/1` che segnano i grandi eventi storici e i mesi di mercato aperto.

---

## 3. FASE 2 – Analisi Esplorativa (EDA) e Analisi degli Shock Storici (Il Cuore Narrativo)

Questa è la parte più affascinante da vedere nei grafici e da raccontare all'esame. Prima di lanciare qualsiasi algoritmo di previsione, **esploriamo i dati per raccontare la storia economica del calcio moderno dal 2012 al 2026**.

### 3.1 La Grande Corsa dei Valori (Trend 2012–2026) e il Confronto tra Campionati
Nel primo blocco di grafici mostreremo:
* Come il valore complessivo delle rose dei Top-5 campionati sia passato da circa **11–12 miliardi di € nel 2012** a oltre **34,5 miliardi di € nel 2026** (quasi **triplicato** in 14 anni!).
* **Il confronto tra i 5 campionati europei (con focus Serie A):**
  * Mostreremo le 5 curve separate (Premier League, LaLiga, Serie A, Bundesliga, Ligue 1).
  * Si vedrà chiaramente come la **Premier League** abbia preso il largo grazie ai diritti TV miliardari, mentre la **Serie A** sia cresciuta da ~2,2 miliardi € fino a circa **5,5 miliardi €** nel 2026, lottando con LaLiga e Bundesliga per il secondo/terzo posto europeo.

### 3.2 Analisi Approfondita dei 3 Grandi Shock Storici (Zoom sugli Eventi)
Proprio come nei progetti d'esempio del corso (che facevano lo zoom sul crollo COVID del 2020 o sugli *stock split* di Apple e Nvidia), noi dedicheremo un'analisi specifica ai **3 grandi shock che hanno cambiato strutturalmente la serie**:

1. **SHOCK 1 – Estate 2017: L'"Effetto Neymar" (222 Milioni €) e l'Esplosione dell'Inflazione**
   * **Cosa è successo nella realtà:** Nell'agosto 2017 il Paris Saint-Germain paga la clausola rescissoria di **222 milioni di €** al Barcellona per Neymar (più 180 milioni per Mbappé). Il Barcellona, con quei soldi in tasca, compra Dembélé e Coutinho a oltre 130–140 milioni ciascuno.
   * **Cosa si vede nei nostri dati:** A partire dall'estate/autunno 2017, la pendenza della curva del valore delle rose **cambia marcia drasticamente**. Un giocatore "normale" che prima del 2017 valeva 15 milioni passa a valerne 30–35. Mostreremo un grafico di zoom sul triennio **2015–2019** che evidenzia come il tasso di crescita annuo prima e dopo l'agosto 2017 abbia subito un vero e proprio salto di livello (*structural break*).

2. **SHOCK 2 – Primavera 2020: Il Crollo COVID-19 (Il "Cigno Nero")**
   * **Cosa è successo nella realtà:** A marzo 2020 il calcio mondiale si ferma per la pandemia. Stadi chiusi, diritti TV congelati e bilanci dei club in profondo rosso. All'inizio di **aprile 2020**, Transfermarkt prende una decisione senza precedenti nella sua storia: applica un **taglio straordinario a tavolino del -20% su quasi tutti i calciatori del mondo** (-10% solo per i giovanissimi nati dopo il 1998).
   * **Cosa si vede nei nostri dati:** Nella nostra serie storica si vede un **crollo verticale nell'aprile 2020** (l'unico vero crollo dell'intero decennio!), seguito da una sessione di mercato estiva 2020 (spostata a settembre-ottobre) con volumi di spesa dimezzati.
   * **Perché questo shock è bellissimo da analizzare:** A differenza degli stadi (che restano a zero per 14 mesi rendendo impossibile usare ARIMA), il valore delle rose subisce un **gradino verso il basso del ~18% in un singolo mese** e poi riprende a muoversi regolarmente! È il caso da manuale perfetto per mostrare l'utilità di una **variabile dummy di intervento (`dummy_covid`)** dentro il modello SARIMAX.

3. **SHOCK 3 – 2022–2026: Il Rimbalzo Post-Pandemia, l'Era dei Maxi-Mercati e il Boom Saudi (2023)**
   * **Cosa è successo nella realtà:** Dal 2022 i ricavi tornano a correre, nell'estate 2023 irrompe la **Saudi Pro League** (e i maxi-investimenti della Premier League come Chelsea, Arsenal, Manchester City) iniettando miliardi di liquidità nei club europei, e dal 2024/25 parte la **nuova Champions League a 36 squadre** con rose ancora più ampie e costose.
   * **Cosa si vede nei nostri dati:** Il valore aggregato non solo recupera i livelli pre-COVID del 2019, ma sfonda il tetto dei **30–34,5 miliardi di €** tra il 2024 e il 2026, accompagnato da sessioni di mercato estive da record (**oltre 4–5 miliardi di € spesi nel solo mese di luglio!**).

### 3.3 Analisi delle Variazioni Percentuali Mensili (`Change %`) e della Volatilità
Esattamente come richiesto nei progetti modello del corso:
* Calcoliamo la serie delle variazioni percentuali mese su mese:
  $$\text{Change \%}_t = \frac{Y_t - Y_{t-1}}{Y_{t-1}} \times 100$$
* **Cosa ci mostra questo grafico:**
  1. Mentre la serie dei valori in miliardi sale sempre, la serie del `Change %` oscilla intorno allo zero, mettendo subito in risalto i **picchi positivi periodici** (le rivalutazioni estive/invernali) e il **grande picco negativo anomalo dell'aprile 2020 (COVID)**.
  2. Tramite istogramma e boxplot del `Change %`, analizziamo la distribuzione delle variazioni mensili e identifichiamo statisticamente i mesi *outlier* (gli shock di mercato).

### 3.4 Relazione tra Valore delle Rose ($Y$) e Volume delle Spese di Mercato ($X$)
Qui prepariamo il terreno per il modello avanzato **SARIMAX**:
* Mettiamo a confronto sullo stesso asse temporale la curva del **Valore delle Rose** con le barre del **Volume Spese Mensile (`transfer_fee` da `transfers.csv`)** e del **Numero di Trasferimenti**.
* Calcoliamo una **Matrice di Correlazione** (sia sullo stesso mese $t$, sia con ritardi di 1–3 mesi) tra:
  * Valore delle Rose e Variazione Mensile (`Change %`);
  * Volume Spese in Milioni € (`transfer_spend_m`);
  * Numero di Trasferimenti (`n_transfers`);
  * Prezzo medio pagato per trasferimento;
  * Rapporto Spesa / Valore di Mercato (*Sovrapprezzo*).
* **Cosa dimostriamo:** Nei mesi (o subito dopo i mesi) in cui i club spendono miliardi sul mercato (Luglio-Agosto e Gennaio), il valore complessivo delle rose registra le sue variazioni più forti. Quindi il `Volume Spese` è un **predittore esogeno naturale** per il nostro modello!

---

## 4. FASE 3 – Come Integriamo l'Analisi Matematica e i Modelli di Time Series (Spiegato Semplice)

Dopo aver fatto una parte di ETL ed EDA ricchissima, passiamo alla pipeline classica delle serie storiche. Ecco cosa fa ogni passaggio spiegato in parole semplici:

### 4.1 Decomposizione Stagionale STL (Smontare la serie in 3 pezzi)
Usiamo l'algoritmo **STL (Seasonal-Trend decomposition using LOESS)** con periodo stagionale **$s = 12$ mesi** per dividere la nostra curva in tre grafici distinti:
1. **Il Trend ($T_t$):** La linea morbida di fondo che mostra la crescita strutturale dal 2012 al 2026 (depurata dai sali-scendi mensili).
2. **La Stagionalità ($S_t$):** L'onda che si ripete ogni 12 mesi. Affiancheremo a questo grafico un **Boxplot per Mese dell'Anno (da Gennaio a Dicembre)** che mostrerà chiaramente come i mesi estivi (Giugno–Settembre) e invernali (Dicembre–Gennaio) abbiano una spinta sistematicamente diversa rispetto ai mesi primaverili/autunnali.
3. **Il Residuo ($R_t$):** Tutto ciò che non è né crescita lenta né ciclo annuale: qui dentro vedremo spuntare come un "fulmine" lo shock del COVID del 2020 e il boom del 2017!

### 4.2 Stazionarietà (Test ADF e KPSS) e Differenziazione
* **Perché serve (spiegato elementare):** I modelli della famiglia ARIMA funzionano bene solo se la serie è **stazionaria**, cioè se non ha un trend che sale all'infinito ma oscilla attorno a una media stabile. La nostra serie parte da 11 miliardi e arriva a 34 miliardi: quindi **non è stazionaria**.
* **Come lo dimostriamo con rigore:** Applichiamo il test statistico **ADF (Augmented Dickey-Fuller)** (e il test complementare **KPSS**) sulla serie originale. Il test ADF darà un *p-value > 0.05*, confermando matematicamente che la serie non è stazionaria.
* **Come la rendiamo stazionaria:** Calcoliamo la **differenza prima** ($d = 1$), cioè invece di guardare il valore assoluto in miliardi ($Y_t$), guardiamo di quanto cambia da un mese all'altro ($\Delta Y_t = Y_t - Y_{t-1}$). Rifacciamo il test ADF sulla serie differenziata: ora il *p-value sarà < 0.001*! La serie è diventata stazionaria e pronta per essere modellata.

### 4.3 Correlogrammi ACF e PACF e Scelta dei Parametri (`auto_arima`)
* Plottiamo i due grafici **ACF (Autocorrelazione)** e **PACF (Autocorrelazione Parziale)** fino a 36 mesi di ritardo.
* **Cosa ci fanno vedere:** Ci mostrano quali mesi del passato influenzano di più il mese presente. Vedremo dei picchi che escono dalle bande di confidenza ai ritardi **12, 24, 36** (la stagionalità annuale di 12 mesi!) e nei primi lag (1, 2).
* Usiamo poi **`auto_arima`** (che testa automaticamente decine di combinazioni minimizzando l'indice **AIC**) per confermare la combinazione ottimale dei parametri $(p, d, q)$ e $(P, D, Q)_{12}$.

### 4.4 La Gara tra i Modelli Predittivi (Train 80% / Test 20%)
Tagliamo la serie in modo cronologico (senza mescolare il futuro col passato, zero *data leakage*):
* **Training Set (80%):** primi ~140 mesi (da Gennaio 2012 a Autunno 2023) su cui i modelli imparano.
* **Test Set (20%):** ultimi ~34 mesi (da fine 2023 a Giugno 2026) su cui valutiamo quanto le previsioni si avvicinano alla realtà.

Metteremo a confronto in progressione:
1. **Baseline (Naive / Seasonal Naive):** Prevede semplicemente l'ultimo valore noto o il trend elementare. Serve come "asticella minima" da battere.
2. **ARIMA $(p, d, q)$:** Modello senza stagionalità. Vedremo sul grafico che tira una linea quasi dritta perché non sa che ogni estate e ogni inverno c'è il calciomercato!
3. **SARIMA $(p, d, q)(P, D, Q)_{12}$:** Aggiunge la stagionalità a 12 mesi. Ora la previsione inizia a seguire le onde annuali del mercato!
4. **SARIMAX $(p, d, q)(P, D, Q)_{12} + \text{Esogene}$:** Il nostro modello di punta. Oltre alla storia passata, gli passiamo come variabili esogene il **Volume Spese di Calciomercato (`transfer_spend_m`)**, il **Numero di Trasferimenti (`n_transfers`)** e la **Dummy Shock COVID**. Mostreremo come l'aggiunta delle spese di mercato migliori sensibilmente la previsione e alzi l'$R^2$!
5. **Confronto con Holt-Winters / Machine Learning (XGBoost):** Aggiungiamo un modello alternativo per rendere la tabella finale di confronto ancora più ricca.

### 4.5 Diagnostica dei Residui e Metriche Finali
* Sul modello migliore (SARIMAX) mostriamo i **4 grafici diagnostici dei residui** (errori nel tempo, istogramma degli errori rispetto alla curva Normale, Q-Q plot, autocorrelazione degli errori) e il **Test di Ljung-Box** per certificare che il modello ha estratto tutta l'informazione utile dai dati.
* Chiudiamo con la **Tabella Comparativa Finale** su **MAE, MSE, RMSE, MAPE e $R^2$**, più l'applicazione diretta anche sul **Focus Serie A**.

---

## 5. Struttura Capitoli e Figure della Parte 2 (Numerazione Continua dopo Spotify)

Visto che la Parte 1 su Spotify si chiude al **Capitolo 5** e alla figura **`fig23`**, la Parte 2 proseguirà dal **Capitolo 6** e dalla figura **`fig24`** mantenendo lo stesso identico stile grafico (300 DPI, stessa palette e formattazione accademica):

* **Capitolo 6 – Football Market Analysis: Introduzione, ETL ed Analisi Esplorativa (EDA)**
  * **6.1 Obiettivi e Scelta del Dominio:** Dal database relazionale di Transfermarkt alla serie storica del Valore di Mercato e Volume Spese.
  * **6.2 Fase di ETL e Data Engineering:** Ricostruzione dinamica mensile delle rose (incrocio valutazioni + trasferimenti + presenze), allineamento temporale ($s=12$, 174 mesi consecutivi) e prevenzione di serie piatte o buchi estivi.
  * **6.3 Analisi Esplorativa (EDA) e Trend di Lungo Periodo (2012–2026):**
    * 📊 `fig24_ts_evoluzione_valore_campionati.png` – Evoluzione storica del Valore di Mercato (Aggregato Top-5 e confronto tra Premier League, Serie A, LaLiga, Bundesliga, Ligue 1).
  * **6.4 Analisi Approfondita degli Shock Storici (Zoom sugli Eventi):**
    * 📊 `fig25_ts_zoom_shock_storici.png` – Pannello in 3 zoom: (1) Shock Neymar e inflazione 2017, (2) Crollo verticale COVID-19 nell'aprile 2020, (3) Boom post-pandemia e Saudi League 2023–2025.
    * 📊 `fig26_ts_variazioni_percentuali_volatilita.png` – Serie storica delle variazioni percentuali mensili (`Change %`), fasce di volatilità e individuazione degli shock anomali.
  * **6.5 Dinamica del Calciomercato: Valore Rose vs Volume Spese (`transfers.csv`):**
    * 📊 `fig27_ts_valore_rose_vs_volume_spese.png` – Confronto temporale tra Valore delle Rose e Volume Spese / Numero Trasferimenti mensili + Matrice di correlazione.

* **Capitolo 7 – Decomposizione Stagionale, Stazionarietà e Analisi di Autocorrelazione**
  * **7.1 Decomposizione STL (Trend, Stagionalità $s=12$, Residuo):**
    * 📊 `fig28_ts_stl_decomposition.png` – Decomposizione STL completa della serie storica.
    * 📊 `fig29_ts_stagionalita_mensile_profilo.png` – Boxplot e profilo stagionale per mese dell'anno (Gennaio–Dicembre): l'effetto delle finestre di mercato e delle revisioni di listino.
  * **7.2 Analisi di Stazionarietà (Test ADF e KPSS) e Differenziazione:**
    * 📊 `fig30_ts_stazionarieta_differenziazione.png` – Serie originale vs Serie differenziata ($\Delta Y_t$) con tabella risultati dei test ADF e KPSS.
  * **7.3 Analisi Spettrale dei Ritardi (ACF e PACF) e Ricerca `auto_arima`:**
    * 📊 `fig31_ts_acf_pacf_correlogrammi.png` – Correlogrammi ACF e PACF fino a 36 lag e selezione degli ordini ottimali tramite AIC/BIC.

* **Capitolo 8 – Modellazione Predittiva: ARIMA, SARIMA, SARIMAX e Confronto Modelli**
  * **8.1 Split Temporale (Train 80% / Test 20%) e Modello Baseline.**
  * **8.2 Modello ARIMA $(p, d, q)$ (Previsione Non Stagionale):**
    * 📊 `fig32_ts_forecast_arima.png` – Previsione ARIMA sul Test Set con intervallo di confidenza al 95%.
  * **8.3 Modello SARIMA $(p, d, q)(P, D, Q)_{12}$ (Previsione Stagionale):**
    * 📊 `fig33_ts_forecast_sarima.png` – Previsione SARIMA sul Test Set con cattura dei cicli annuali.
  * **8.4 Modello SARIMAX con Variabili Esogene (Volume Spese, N. Trasferimenti, Shock COVID):**
    * 📊 `fig34_ts_forecast_sarimax.png` – Previsione SARIMAX sul Test Set e miglioramento predittivo grazie ai volumi di spesa del calciomercato.
  * **8.5 Confronto Generale con Modelli Alternativi (Holt-Winters & XGBoost) e Diagnostica dei Residui:**
    * 📊 `fig35_ts_confronto_modelli_testset.png` – Confronto diretto di tutte le curve di previsione sul Test Set (2023–2026).
    * 📊 `fig36_ts_diagnostica_residui_sarimax.png` – I 4 grafici diagnostici dei residui del modello SARIMAX + Test di Ljung-Box.
    * 📊 `fig37_ts_metriche_confronto_finale.png` – Confronto grafico delle metriche di errore (MAE, RMSE, MAPE, $R^2$) tra tutti i modelli.

* **Capitolo 9 – Focus Specifico sulla Serie A e Conclusioni**
  * **9.1 Analisi e Previsione SARIMAX sul Campionato di Serie A:**
    * 📊 `fig38_ts_focus_serie_a.png` – Serie storica, shock e previsione SARIMAX specifica per il valore delle rose di Serie A.
  * **9.2 Conclusioni e Risposta ai Quesiti di Ricerca.**

