# Guida Completa e Semplicissima alla Parte 2 (Time Series - Transfermarkt)

> **Obiettivo di questo documento**: Spiegare **tutto** quello che è stato fatto nella Parte 2 del progetto di Data Science, passo dopo passo, con parole così semplici ed esempi così chiari che anche un ragazzino delle scuole medie (o chiunque non abbia mai studiato le serie storiche) possa capire ogni singolo grafico, formula, tabella e scelta progettuale — ed essere prontissimo a spiegarlo all'esame!

---

## Indice della Guida
1. [L'idea di base: cosa stiamo studiando e perché?](#1-lidea-di-base-cosa-stiamo-studiando-e-perché)
2. [Capitolo 6: Dai dati grezzi alla serie storica pulita (Il processo di ETL)](#2-capitolo-6-dai-dati-grezzi-alla-serie-storica-pulita-il-processo-di-etl)
3. [Capitolo 7: Esploriamo la storia del calcio e i 3 grandi "terremoti" (EDA)](#3-capitolo-7-esploriamo-la-storia-del-calcio-e-i-3-grandi-terremoti-eda)
4. [Capitolo 8: Apriamo il motore della serie (Decomposizione STL, Stazionarietà e Memoria)](#4-capitolo-8-apriamo-il-motore-della-serie-decomposizione-stl-stazionarietà-e-memoria)
5. [Capitolo 9: La gara tra i modelli predittivi, il focus Serie A e le conclusioni](#5-capitolo-9-la-gara-tra-i-modelli-predittivi-il-focus-serie-a-e-le-conclusioni)
6. [Come è organizzato il codice Python (per spiegarlo al volo ai professori)](#6-come-è-organizzato-il-codice-python-per-spiegarlo-al-volo-ai-professori)
7. [Domande tipiche d'esame con la risposta pronta](#7-domande-tipiche-desame-con-la-risposta-pronta)

---

## 1. L'idea di base: cosa stiamo studiando e perché?

### Cos'è una "Serie Storica" (*Time Series*)?
Immagina di avere l'album delle figurine dei calciatori, ma invece di guardare solo quanto vale un giocatore *oggi*, tieni un **diario mensile** per 15 anni di fila (da **giugno 2011** a **maggio 2026**, cioè **180 mesi**):
- Il 1° giugno 2011 sommi il prezzo di tutti i giocatori dei 5 campionati più importanti d'Europa (**Serie A** italiana, **Premier League** inglese, **LaLiga** spagnola, **Bundesliga** tedesca, **Ligue 1** francese) e scrivi il totale sul diario.
- Il 1° luglio 2011 fai la stessa cosa.
- Vai avanti così, mese dopo mese, senza saltare neanche un mese, fino a maggio 2026.

Questa fila ordinata di 180 numeri (uno per ogni mese) è una **serie storica**.

### Perché abbiamo scelto il "Valore delle Rose" (e le spese di mercato) invece dei gol o degli spettatori allo stadio?
All'inizio ci eravamo chiesti se studiare gli spettatori allo stadio o i gol segnati. Seguendo le indicazioni del dott. Buratti, abbiamo scelto di concentrarci **in verticale sul valore di mercato delle rose e sul calciomercato** per un motivo matematico fondamentale:
- **Le partite e gli spettatori si fermano d'estate**: a giugno e luglio non si gioca! Quindi la serie degli spettatori avrebbe dei "buchi" ogni estate (oppure una sola partita strana di recupero, come Atalanta-Fiorentina giocata il 2 giugno 2024, farebbe sembrare che a giugno 2024 gli stadi fossero pienissimi!). Inoltre, durante il COVID gli stadi sono rimasti a **zero spettatori per 14 mesi di fila**.
- **Il valore economico delle rose invece esiste 12 mesi su 12**: una squadra di calcio ha un valore patrimoniale sia a dicembre (mentre gioca), sia a luglio e agosto (mentre compra e vende giocatori al calciomercato). Così otteniamo una serie storica **continua, senza buchi e lunga esattamente 180 mesi**.

---

## 2. Capitolo 6: Dai dati grezzi alla serie storica pulita (Il processo di ETL)

Il dataset scaricato da Kaggle (*Football Data from Transfermarkt*) **non è** una tabella già pronta mese per mese. È un database diviso in 12 file `.csv` diversi. Noi abbiamo usato i **5 file fondamentali** (riassunti nella **Tabella 1** della relazione):

1. `player_valuations.csv` (**656.301 righe**): ogni riga dice *"Il giorno X, il calciatore Y è stato valutato Z euro"*.
2. `transfers.csv` (**175.165 righe**): ogni riga dice *"Il giorno X, il calciatore Y è passato dalla squadra A alla squadra B per W milioni di euro"*.
3. `appearances.csv` (**1.894.350 righe**): ogni riga è una presenza in campo di un giocatore in una partita, con scritto per quale squadra stava giocando in quel giorno preciso.
4. `clubs.csv` (**796 squadre**): dice a quale campionato appartiene ogni squadra.
5. `competitions.csv` (**65 competizioni**): ci serve per prendere solo i 5 campionati Top (`IT1` Serie A, `GB1` Premier, `ES1` Liga, `L1` Bundesliga, `FR1` Ligue 1).

### I 2 grandi "trabocchetti" dei dati grezzi (e come li abbiamo risolti nell'ETL)
Se avessimo semplicemente sommato i numeri di `player_valuations.csv` mese per mese, sarebbe uscito un disastro per colpa di **due trabocchetti nascosti**:

#### Trabocchetto 1: "I listini non vengono aggiornati tutti i mesi!" (`Figura 1`)
- Gli esperti di Transfermarkt non cambiano il prezzo di tutti i 3.500 giocatori ogni singolo mese: fanno i grandi aggiornamenti dei prezzi a **marzo, giugno, ottobre e dicembre**.
- E a **luglio, agosto e gennaio**? In quei mesi fanno pochissimi aggiornamenti di listino perché aspettano che finisca il calciomercato! Però proprio a luglio, agosto e gennaio i club comprano e vendono migliaia di giocatori (spendendo in media **2,3 miliardi a luglio**, **1,28 miliardi ad agosto** e **550 milioni a gennaio**, come si vede nella **Figura 1**).
- **La nostra soluzione**:
  1. Per ogni giocatore in attività, manteniamo valido il suo ultimo prezzo conosciuto da un mese all'altro (operazione chiamata *forward-fill*, con un limite massimo di 18 mesi così se un giocatore si ritira dal calcio smette automaticamente di essere contato).
  2. Contemporaneamente, ogni mese aggiorniamo le squadre con i giocatori appena comprati o venduti nel calciomercato (`transfers.csv`). Così la nostra serie si muove in modo realistico **in tutti i 180 mesi**!

#### Trabocchetto 2: "L'errore della macchina del tempo (`current_club_id`)" (`Figura 2` e `Listato 1`)
- Dentro `player_valuations.csv` c'è una colonna chiamata `current_club_id` (squadra attuale).
- **Attenzione**: quella colonna non dice in quale squadra giocava il calciatore nel 2015 o nel 2019, ma dice **solo in quale squadra gioca oggi nel 2026**!
- *Esempio pratico*: se un campione giocava nella Juventus o nell'Inter nel 2019, ma nel 2023 è andato a giocare in Arabia Saudita o si è ritirato, la colonna `current_club_id` dice "Arabia Saudita". Se facessimo la somma usando quella colonna, **tutti i suoi milioni del 2019 sparirebbero dalla Serie A!**
- **La nostra soluzione (`Listato 1`)**: Abbiamo incrociato mese per mese le partite giocate (`appearances.csv`) e i trasferimenti (`transfers.csv`) per ricostruire **la vera squadra in cui ogni giocatore militava in quel mese preciso**.
- **La prova visiva (`Figura 2`)**: Nel grafico superiore della Figura 2 mostriamo la differenza tra il calcolo "ingenuo" (linea arancione tratteggiata) e la nostra ricostruzione storica (linea verde): il calcolo ingenuo perdeva per strada quasi **1 miliardo di euro** di valore della Serie A nel 2019–2020!

### Perché ci siamo fermati a Maggio 2026 (180 mesi)?
Il dataset è stato estratto a metà giugno 2026 (`12 giugno 2026`): il mese di giugno 2026 era quindi incompleto e conteneva già le scadenze automatiche dei contratti del 30 giugno. Fermandoci a **maggio 2026** otteniamo **180 mesi perfetti** (da `2011-06` a `2026-05`), cioè **15 anni esatti** ($15 \times 12 = 180$), senza neanche un dato mancante (**Tabella 2**).

---

## 3. Capitolo 7: Esploriamo la storia del calcio e i 3 grandi "terremoti" (EDA)

Una volta ottenuta la tabella pulita (`market_monthly.csv`), nel Capitolo 7 guardiamo cosa è successo in questi 15 anni con **9 figure grandi e nitide** (`fig03` .. `fig11`).

### A. Il confronto tra i 5 campionati: la fuga della Premier League (`Figura 3`, `Figura 4`, `Tabella 3`)
In 15 anni il valore totale delle rose dei 5 campionati è passato da **11,12 miliardi di euro** (giugno 2011) a **34,44 miliardi di euro** (maggio 2026), crescendo del **+209,7%** (più del triplo!). Ma chi è cresciuto di più?
- **Nel 2011 erano tutti vicini**:
  - **Premier League**: 3,15 miliardi (il **28,3%** della torta europea)
  - **LaLiga**: 2,58 miliardi (23,2%)
  - **Serie A**: 2,30 miliardi (**20,7%**)
  - **Bundesliga**: 1,65 miliardi (14,9%)
  - **Ligue 1**: 1,45 miliardi (13,0%)
- **Nel 2026 la Premier League ha fatto il vuoto (`Figura 4`)**:
  - Grazie ai diritti TV venduti in tutto il mondo a cifre enormi, la Premier League è cresciuta del **+330,6%**, arrivando a **13,55 miliardi** (con un picco di **13,87 miliardi** a novembre 2025): oggi da sola vale quasi il **40% (39,3%)** di tutto il calcio europeo!
  - **E la Serie A?** È rimasta quasi ferma tra il 2011 e il 2015 (intorno a 2,3–2,9 miliardi), poi è cresciuta tantissimo tra il 2016 e il marzo 2020 toccando il suo **record storico di 6,08 miliardi di euro**. Dopo la botta del COVID si è stabilizzata: a maggio 2026 vale **5,40 miliardi** (+134,9% rispetto al 2011, pari al **15,7%** della torta europea), al 3° posto dietro Premier (13,55 Mld) e Liga spagnola (5,72 Mld) e davanti alla Bundesliga (5,29 Mld).

### B. I 3 grandi "terremoti" (Shock storici) della serie
Guardando la curva non si vede una linea dritta, ma si vedono **3 svolte clamorose**:

1. **Shock 1 – Agosto 2017: L'"Effetto Neymar" (`Figura 5`)**
   - Nell'estate 2017 il Paris Saint-Germain paga la clausola record di **222 milioni di euro** al Barcellona per comprare Neymar (e subito dopo il Barcellona spende quei soldi per Dembélé e Coutinho, mentre il PSG prende anche Mbappé per 180 milioni).
   - Quella pioggia di denaro ha "gonfiato" i prezzi di tutti i giocatori d'Europa: prima di agosto 2017 il valore totale saliva piano (**+1,31 miliardi all'anno**); dopo agosto 2017 la pendenza si impenna a **+5,46 miliardi all'anno** (4 volte più veloce!).

2. **Shock 2 – Aprile 2020: Il crollo del COVID-19 (`Figura 6`)**
   - A marzo 2020 scoppia la pandemia, si fermano tutti i campionati e chiudono gli stadi.
   - Ad **aprile 2020** Transfermarkt decide una misura mai vista nella storia: taglia a tavolino il prezzo di quasi tutti i calciatori del mondo (-20% per i giocatori normali, -10% per i giovanissimi), facendo **4.379 ribassi in un mese solo**.
   - Il valore europeo crolla in 30 giorni da 33,79 a 27,65 miliardi: **persi 6,14 miliardi di euro in un mese (-18,17%)**! Anche la Serie A perde il **-18,34%** (da 6,08 a 4,97 miliardi).
   - Quanto tempo ci è voluto per riprendersi da quella botta? Esattamente **55 mesi** (4 anni e mezzo!): il tetto dei 33,79 miliardi è stato superato di nuovo solo nell'**ottobre 2024** (33,83 miliardi).

3. **Shock 3 – 2023–2026: L'Arabia Saudita, la nuova Champions e i super-campioni (`Figura 7`)**
   - Dall'estate 2023 entrano nel mercato i ricchissimi club della **Saudi Pro League** (che comprano giocatori dall'Europa a caro prezzo, dando ai club europei soldi freschi da reinvestire) e nel 2024/2025 parte la nuova **Champions League a 36 squadre** con premi UEFA più alti.
   - Il mercato tocca il massimo storico di **34,44 miliardi** a maggio 2026, e il prezzo medio di un singolo calciatore (`mean_player_value_m`), che fino al 2017 era di circa 3 milioni, schizza a **11,01 milioni di euro**!

### C. Le variazioni percentuali mese per mese e gli "Outlier" (`Figura 8` e `Figura 9`)
Se invece di guardare i miliardi totali guardiamo **di quale percentuale ($\Delta\%$) cambia il mercato ogni mese**:
- In un mese normale il mercato sale in media dello **+0,66%** ($\mu$), oscillando quasi sempre tra **-3,46% e +4,78%** (le due linee tratteggiate della fascia $\mu \pm 2\sigma$).
- I pallini rossi fuori da questa fascia sono i **mesi anomali (*outlier*)**:
  - Il picco positivo più alto è **gennaio 2018 (+9,24%)**, quando i listini vengono adeguati al rialzo dopo l'estate di Neymar.
  - Il picco negativo più estremo è **aprile 2020 (-18,17%)**, che dista più di 9 deviazioni standard dalla media ($z = -9,14$).
- Nella **Figura 9** (l'istogramma con la curva di densità KDE) questo crollo crea una "coda lunga" a sinistra (in statistica si chiama **asimmetria negativa**, *skewness* $= -3,69$) e un picco molto stretto al centro con code pesanti (**leptocurtosi**, *kurtosis* $= 35,48$).

### D. Più i club spendono al calciomercato, più sale il valore delle rose? (`Figura 10` e `Figura 11`)
- In 15 anni i club dei 5 campionati hanno fatto **31.000 trasferimenti**, spendendo in totale **65,82 miliardi di euro** (di cui **14,68 miliardi** spesi dalla Serie A, cioè il 22,3% di tutta la spesa europea!).
- Nella **Figura 10** vediamo che i soldi vengono spesi quasi tutti in estate (**luglio e agosto**, con il record storico di **4,46 miliardi spesi nel solo luglio 2025**) e un po' a **gennaio**.
- Nella **Figura 11 (Matrice di correlazione)** facciamo una scoperta preziosissima per il Capitolo 9:
  - Se confrontiamo il valore delle rose con la spesa di *quel singolo mese*, la correlazione è bassa (**0,24**), perché per 9 mesi all'anno il mercato è chiuso e la spesa è zero mentre le rose continuano a valere più di 30 miliardi!
  - Se invece sommiamo **la spesa degli ultimi 12 mesi (`Spesa mobile 12m`)**, la correlazione col valore delle rose vola a **0,88 (88%)**!
  - *Cosa significa?* Che i soldi veri spesi nel calciomercato nell'ultimo anno sono il "carburante" che fa salire il valore delle squadre. Ecco perché nel Capitolo 9 useremo questa variabile per aiutare il modello SARIMAX!

---

## 4. Capitolo 8: Apriamo il motore della serie (Decomposizione STL, Stazionarietà e Memoria)

Prima di dare i dati in pasto ai modelli predittivi, dobbiamo "prepararli" matematicamente. Il Capitolo 8 fa tre operazioni fondamentali (`fig12` .. `fig15`).

### Passo 1: Smontiamo la curva in 3 pezzi con la Decomposizione STL (`Figura 12` e `Figura 13`)
La tecnica **STL** prende la nostra curva $Y_t$ e la divide nella somma di 3 pezzi ($Y_t = T_t + S_t + R_t$):
1. **Trend ($T_t$)**: è la salita di fondo negli anni. Ha una forza pari a **$F_T = 0,992$** (il massimo possibile è 1,0!): significa che il 99% del movimento di lungo periodo è guidato dalla grande crescita economica del calcio.
2. **Stagionalità ($S_t$)**: è il battito che si ripete ogni 12 mesi ($s = 12$). Guardando la **Figura 13** (la media di ogni mese dell'anno), scopriamo quali sono i mesi "ricchi" e quelli "magri":
   - **Ottobre (+0,28 miliardi)** e **Novembre (+0,15 miliardi)** sono i mesi più alti, perché dopo la chiusura del mercato estivo Transfermarkt aggiorna i listini di tutti i campionati rialzando i nuovi acquisti e i protagonisti delle coppe, preceduti da **Luglio (+0,09 Mld)** e **Settembre (+0,08 Mld)**.
   - **Maggio (-0,10 Mld)** e **Giugno (-0,12 Mld)** (così come gennaio/febbraio) hanno invece un effetto medio negativo perché i giocatori invecchiano di un anno e scadono i contratti a fine stagione.
3. **Residuo ($R_t$)**: è tutto ciò che non è né crescita normale né stagione. Infatti nella Figura 12 il residuo sta quasi sempre piatto sullo zero, tranne quando c'è il "buco" improvviso del COVID ad aprile 2020!

### Passo 2: La serie "sta ferma"? I test ADF e KPSS e la Differenziazione (`Figura 14` e `Tabella 4`)
- **Il problema**: I modelli matematici come ARIMA non possono lavorare su una curva che sale sempre verso l'alto (da 11 a 34 miliardi). Hanno bisogno di una serie **stazionaria**, cioè che oscilla su e giù attorno a una linea orizzontale (come le onde del mare attorno al livello zero).
- **Come lo dimostriamo ai professori?** Usiamo due test statistici famosissimi che fanno da "giudici" (**Tabella 4**):
  1. **Test ADF (Augmented Dickey-Fuller)**: fa il tifo per la *non-stazionarietà*. Dice che la serie è stazionaria solo se il suo `p-value` è **piccolissimo ($< 0,05$)**.
  2. **Test KPSS**: funziona esattamente al contrario! Dice che la serie è stazionaria solo se il suo `p-value` è **grande ($> 0,05$)**.
- **Il verdetto dei due giudici**:
  - Sulla **serie originale ($d = 0$)**: ADF dà `p-value = 0,8148` (troppo alto!) e KPSS dà `p-value = 0,0100` (troppo basso!). Entrambi dicono: **"Bocciata! Non è stazionaria"**.
  - Allora facciamo un trucco semplicissimo chiamato **Differenziazione prima ($d = 1$)**: invece di prendere il valore totale del mese, calcoliamo **la differenza rispetto al mese scorso** ($\Delta Y_t = Y_t - Y_{t-1}$, cioè di quanto è salito o sceso in quel mese).
  - Sulla **serie differenziata ($d = 1$)**: ADF dà `p-value < 0,0001` e KPSS dà `p-value = 0,1000`. Entrambi i giudici dicono: **"Promossa! Ora è stazionaria!"** (come si vede nel pannello verde in basso della **Figura 14**).
  - *Risultato*: nei modelli ARIMA metteremo il parametro **$d = 1$**.

### Passo 3: Quanta "memoria" ha il mercato? I grafici ACF e PACF (`Figura 15`)
Ora che abbiamo la serie differenziata che sta ferma attorno allo zero, ci chiediamo: *per indovinare il mese prossimo, a quanti mesi indietro dobbiamo guardare?*
Costruiamo due grafici a bastoncini fino a 36 mesi indietro (**Figura 15**):
- **ACF (Autocorrelazione)** e **PACF (Autocorrelazione Parziale)**.
- Cosa notiamo?
  1. Nei primi mesi (ritardo 1, 2, 3) i bastoncini sono bassi perché l'enorme salto del COVID (-6,14 miliardi in un solo mese) "schiaccia" matematicamente la correlazione dei mesi normali (*effetto di mascheramento dell'outlier*). Questo ci dice di usare ordini piccoli e semplici per il breve periodo ($p = 1, q = 1$).
  2. Al **ritardo 12 (linea tratteggiata arancione a 12 mesi)** c'è un chiaro picco positivo (`+0,107`): quello che succede in un mese somiglia a quello che è successo **nello stesso mese dell'anno precedente**! Questo conferma che serve un modello stagionale con ciclo di 12 mesi ($s = 12$).

---

## 5. Capitolo 9: La gara tra i modelli predittivi, il focus Serie A e le conclusioni

Siamo arrivati al gran finale: mettere alla prova i modelli matematici per vedere chi indovina meglio il valore del mercato!

### A. Come si fa una gara onesta tra modelli? La divisione 80/20 (`Figura 16`)
Non possiamo barare facendo vedere ai modelli i dati del 2025 prima di chiedergli di indovinare il 2025! Quindi tagliamo i nostri 180 mesi in due pezzi ordinati nel tempo (**Figura 16**):
1. **Training Set (I primi 144 mesi = 12 anni esatti, da giugno 2011 a maggio 2023, l'80%)**: sono i "libri di scuola" su cui i modelli studiano e imparano le regole del mercato.
2. **Test Set (Gli ultimi 36 mesi = 3 anni esatti, da giugno 2023 a maggio 2026, il 20%)**: li teniamo nascosti, chiediamo ai modelli di prevederli e poi confrontiamo le loro previsioni con i dati veri!

### B. Come diamo i voti ai modelli? Le 3 metriche di errore
Per sapere di quanto sbaglia un modello sui 36 mesi del Test Set, calcoliamo:
- **MAE (Errore Assoluto Medio)**: di quanti miliardi di euro sbaglia in media ogni mese (più è vicino a 0, più è bravo).
- **RMSE (Errore Quadratico Medio)**: simile al MAE, ma siccome eleva gli errori al quadrato prima di fare la media, "punisce" severamente chi fa anche solo uno o due errori grossi.
- **MAPE (Errore Percentuale Medio)**: di quale **percentuale (%)** sbaglia in media (es. sbagliare dell'1% su 33 miliardi è pochissimo!).

### C. I 5 modelli in gara e chi vince (`Figura 17`, `Figura 18`, `Tabella 5`)
Abbiamo messo in gara 5 modelli dal più semplice al più intelligente:

| Posizione | Modello in gara | Come ragiona (spiegato semplice) | Errore RMSE (Mld €) | Errore MAE (Mld €) | Errore MAPE (%) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 5° | **Baseline Naive** | *Il modello pigro*: dice che da giugno 2023 al 2026 il valore resterà sempre fermo a quello di maggio 2023 (31,01 Mld). | 2,078 | 1,877 | **5,64%** |
| 4° | **ARIMA(1, 1, 1)** | Capisce che il mercato sale nel tempo e guarda il mese precedente, ma **non conosce le stagioni** (tira una curva liscia). | 0,792 | 0,650 | **1,95%** |
| 3° | **Holt-Winters ($s=12$)** | Usa medie pesate (smoothing esponenziale) per seguire sia la salita sia il ciclo dei 12 mesi. | 0,797 | 0,625 | **1,88%** |
| 2° | **SARIMA(1,1,1)$\times$(1,0,1)$_{12}$** | Unisce l'ARIMA al ciclo annuale di 12 mesi: segue bene le onde stagionali ma guarda solo il passato dei prezzi. | 0,703 | 0,476 | **1,45%** |
| 🏆 **1°** | **SARIMAX + Esogene** | **Il campione**: oltre a tutto quello che ha il SARIMA, legge anche **quanti soldi veri i club hanno speso nel calciomercato negli ultimi 12 mesi**, il flusso di trasferimenti e lo shock COVID! | **0,524** | **0,382** | **1,15%** |

Guardando la **Figura 17** e la **Figura 18**:
- Passare dal modello pigro (Naive, errore 5,64%) al **SARIMAX con le spese di calciomercato** abbatte l'errore di quasi il **75%**, scendendo a un errore percentuale medio di appena l'**1,15%**!
- Nella Figura 17 si vede che la linea verde del SARIMAX segue passo passo tutti i saliscendi della linea nera reale dal 2023 al 2026.

### D. Il controllo qualità: i residui e il Test di Ljung-Box (`Figura 19`, `Figura 20`, `Tabella 6`)
Dopo che un modello ha vinto, un bravo data scientist fa un ultimo controllo: guarda gli **errori rimasti** (chiamati **residui**, cioè la differenza tra realtà e modello) per verificare che siano **Rumore Bianco (*White Noise*)**.
- *Cosa vuol dire Rumore Bianco?* Vuol dire che i piccoli errori che il modello commette sono puramente casuali (come il lancio di una moneta), centrati sullo zero e con la forma a campana della curva Normale (**Figura 19** e **Figura 20**). Se non fossero casuali, vorrebbe dire che il modello si è dimenticato qualche regola importante!
- **Il Test di Ljung-Box (`Tabella 6`)**: è il test matematico che controlla se i residui sono davvero casuali. Qui la regola è: **il `p-value` deve essere maggiore di 0,05 ($p > 0,05$)**.
- Nel nostro modello SARIMAX, a 6, 12 e 24 mesi il `p-value` è sempre altissimo (**tra 0,56 e 0,59**, molto sopra 0,05): **esame superato a pieni voti!**

### E. Il Focus sulla Serie A italiana (`Figura 21`)
Infine, abbiamo preso il nostro modello campione **SARIMAX** e lo abbiamo applicato solo alla **Serie A italiana** (usando come aiuto esterno i soldi spesi nel calciomercato dai club di Serie A e il numero di acquisti in Italia):
- La previsione arancione tratteggiata nella **Figura 21** ricalca quasi alla perfezione la curva nera reale della Serie A dal 2023 al 2026!
- L'errore medio scende a un **MAPE dell'1,25%** (contro il 3,90% della Baseline Naive), che corrisponde a un errore medio assoluto (**MAE**) di appena **69 milioni di euro** (0,069 miliardi) su un campionato che vale più di 5.400 milioni di euro!

---

## 6. Come è organizzato il codice Python (per spiegarlo al volo ai professori)

Se all'esame ti chiedono *"Come avete strutturato il codice della Parte 2?"*, puoi mostrare con orgoglio che non avete fatto un file unico confuso, ma un **package Python modulare e pulito** dentro `Parte_2_Time_Series/`:

```text
Parte_2_Time_Series/
├── src/                               # IL MOTORE MODULARE (Funzioni riutilizzabili)
│   ├── config.py                      # Tutte le cartelle, le date e i colori in un posto solo
│   ├── plotting.py                    # Funzioni per salvare i grafici HD a 300 DPI con font grandi
│   ├── analysis/
│   │   └── stl_stationarity.py        # Funzioni per STL, test ADF/KPSS e grafici ACF/PACF (Cap. 8)
│   └── modeling/
│       └── sarimax_models.py          # Funzioni per Naive, ARIMA, Holt-Winters, SARIMA, SARIMAX e Serie A (Cap. 9)
│
├── scripts/                           # GLI SCRIPT DA LANCIARE
│   ├── 01_etl_transfermarkt.py        # Pulisce i CSV grezzi e crea market_monthly.csv (Cap. 6)
│   ├── 02_eda_e_shock.py              # Crea i grafici esplorativi e degli shock (Cap. 7)
│   ├── 03_run_stl_stationarity.py     # Lancia i calcoli e i grafici del Capitolo 8
│   ├── 04_run_modeling.py             # Lancia i modelli e i grafici del Capitolo 9
│   └── run_all_pipeline.py            # UNICO COMANDO che esegue tutto in automatico!
```

---

## 7. Domande tipiche d'esame con la risposta pronta

1. **"Perché non avete aggregato direttamente `player_valuations.csv` usando la colonna `current_club_id`?"**
   - *Risposta*: Perché `current_club_id` è un campo statico che riporta solo il club in cui il giocatore milita nel 2026. Usandolo direttamente avremmo attribuito tutta la carriera passata di un giocatore al suo campionato del 2026, sottostimando ad esempio la Serie A del 2019–2020 di quasi 1 miliardo di euro. Per evitarlo abbiamo ricostruito il vero club storico mese per mese incrociando `appearances.csv` e `transfers.csv`.

2. **"Perché avete usato sia il test ADF sia il test KPSS per la stazionarietà?"**
   - *Risposta*: Perché hanno ipotesi nulle opposte e si completano a vicenda: l'ADF ha come ipotesi nulla la *non-stazionarietà* (radice unitaria), mentre il KPSS ha come ipotesi nulla la *stazionarietà*. Quando entrambi concordano (come sulla nostra serie differenziata prima con $d = 1$, dove ADF ha $p < 0,0001$ e KPSS ha $p = 0,10$), siamo certi al 100% che la serie è stazionaria.

3. **"Perché il modello SARIMAX batte il modello SARIMA normale?"**
   - *Risposta*: Perché il valore delle rose non si muove solo per inerzia del passato, ma risponde agli investimenti reali dei club e agli shock esterni. Aggiungendo come variabili esogene la spesa cumulata mobile a 12 mesi nel calciomercato (che ha una correlazione di $r = 0,88$ col valore delle rose) e il controllo per lo shock COVID di aprile 2020, l'errore MAPE scende dall'1,45% del SARIMA all'1,15% del SARIMAX (e all'1,25% sulla Serie A).

