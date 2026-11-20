# Guida Completa e Semplicissima alla Parte 2 (Time Series – Transfermarkt)

> **Obiettivo di questo documento:** spiegare in modo chiaro, discorsivo e fedele al 100% alla relazione definitiva (`Relazione/main.pdf`) e al codice Python tutto il lavoro della Parte 2 (Transfermarkt), con i numeri reali verificati e le risposte pronte per l'esame orale.

---

## 1. L'idea di base: cosa studiamo e perché?

Costruiamo una **serie storica mensile regolare** di **156 mesi consecutivi** (13 cicli annuali completi, da **giugno 2013 a maggio 2026**, zero mesi mancanti) del **valore di mercato complessivo delle rose** dei cinque maggiori campionati europei (**Premier League**, **LaLiga**, **Serie A**, **Bundesliga**, **Ligue 1**), affiancata da un focus sulla **Serie A** e dai volumi reali di spesa per **soli acquisti in entrata**.

**Perché il valore delle rose invece di gol o spettatori?**
- **Gol e cartellini:** quasi costanti nel tempo (serie piatta, rumore bianco).
- **Spettatori allo stadio:** buco estivo a giugno–luglio, anomalie di calendario e **14 mesi consecutivi a zero** durante il COVID-19.
- **Valore di mercato delle rose:** esiste tutti i 12 mesi dell'anno, ha una dinamica economica ricca (quasi triplicata in 13 anni) e si presta perfettamente all'analisi con STL, test di stazionarietà e modelli ARIMA/SARIMA/SARIMAX.

---

## 2. Sezione 1 (`capitolo6.tex`): ETL causale e controllo di copertura

Il dataset Kaggle (*Football Data from Transfermarkt*) è un database relazionale di 12 CSV. La pipeline [01_etl_transfermarkt.py](scripts/01_etl_transfermarkt.py) integra **7 tabelle** (`player_valuations.csv` 656.301 righe, `transfers.csv` 175.165, `appearances.csv` 1.894.350, `games.csv` 88.958, `clubs.csv` 796, `competitions.csv` 65, `national_teams.csv` 124) risolvendo tre problemi chiave in modo **puramente causale** (zero `bfill`, zero *look-ahead bias*):

1. **Complementarità tra listini e mercato (`Figura 1`):** gli aggiornamenti dei prezzi si concentrano a marzo, giugno, ottobre e dicembre, mentre gli acquisti reali avvengono a **luglio (2.176 Mln €/mese in media)**, **agosto (1.205 Mln €)** e **gennaio (547 Mln €)**. Le valutazioni individuali vengono quindi propagate in avanti (`ffill`) per un massimo di **15 mesi**.
2. **Eliminazione della doppia distorsione statica (`Figura 2`):**
   - `current_club_id` (in `player_valuations.csv`) e `domestic_competition_id` (in `clubs.csv`) fotografano solo il 2026: usarli distorce il passato (per la Serie A fino a **0,34 miliardi di euro** di sottostima nell'ottobre 2019).
   - Ricostruiamo il club mese per mese prendendo l'evento cronologicamente più recente tra **presenze di club** (valide **12 mesi**, dopo aver escluso le gare per Nazionali tramite `competitions.csv` e `national_teams.csv`) e **trasferimenti** (validi fino a **36 mesi**, partendo dal *warm-up* 2009 per preservare le rose dei club neopromossi non coperti in seconda divisione da `appearances.csv`).
   - Il campionato effettivo del club nella singola stagione è ricavato da `games.csv` (gestendo promozioni, retrocessioni e l'override fattuale di **luglio 2020** nella stagione 2019/2020, in cui furono giocate 211 gare post-lockdown).
3. **Stabilità di copertura (`Figura 2b`):** i **15.591 rientri gratuiti da prestito** del 30 giugno vengono posticipati di un giorno al 1° luglio (nuova stagione) per evitare il picco artificiale di fine giugno. La copertura rimane stabile su tutti i 13 anni senza derive: in media **2.695 calciatori attivi** sui Top-5 (**27,6 giocatori per club** su una media effettiva di 97,55 club, tenendo conto del passaggio della Ligue 1 da 20 a 18 squadre dal 2023/2024) e **582 in Serie A** (29,1 per club).

---

## 3. Sezione 2 (`capitolo7.tex`): Analisi esplorativa, shock e correlazione spuria

- **Crescita 2013–2026 (`Figure 3–4`, `Tabella 3`):** il mercato Top-5 passa da **11,69** a **33,19 miliardi di euro** (**+183,9%**); il valore medio per calciatore attivo sale da **4,42** a **12,62 milioni** (+185,6%) e il mediano triplica da **2,00** a **6,00 milioni**.
  - **Premier League:** da 3,40 a **12,67 Mld** (**+272,9%**, picco 12,78 Mld a ottobre 2025; quota di mercato dal **29,1%** al **38,2%**).
  - **Serie A:** da 2,25 al picco storico di **5,57 Mld** (marzo 2020), poi **5,33 Mld** a maggio 2026 (**+137,3%**, quota dal **19,2%** al **16,1%**, terzo posto dietro Premier e LaLiga 5,64 Mld).
- **Tre fasi storiche (`Figure 5–7`):**
  1. **Accelerazione 2017–2020 (effetto Neymar, `Figura 5`):** la pendenza lineare passa da **+1,13 Mld/anno** (2013–2017) a **+5,47 Mld/anno** (2017–2020, $4{,}84\times$).
  2. **Shock COVID-19 (`Figura 6`):** l'8 aprile 2020 Transfermarkt taglia i listini del $-20\%$ ($-10\%$ per gli under-22): i Top-5 perdono **$-18,59\%$** (**$-5,71$ Mld**, da 30,69 a 24,99 Mld), la Serie A perde **$-18,93\%$** (da 5,57 a 4,52 Mld). Il livello pre-COVID viene recuperato per la prima volta dopo **60 mesi esatti** (marzo 2025, 30,75 Mld) e consolidato stabilmente dall'agosto 2025 (30,81 Mld, 65 mesi).
  3. **Espansione 2023–2026 (`Figura 7`):** liquidità Saudi Pro League + nuova Champions League a 36 squadre portano il massimo a **33,19 Mld** (maggio 2026).
- **Variazioni mensili e outlier (`Figure 8–9`):** media $+0,71\%$, $\sigma = 2,49\%$; mediana $+0,28\%$, $\mathrm{MAD}_n = 1,01\%$; skewness $-2,26$, curtosi in eccesso $24,27$. Aprile 2020 ($-18,59\%$) dista $-7,75\sigma$ classici e **$-18,67$ unità MAD** robuste. Il massimo rialzo reale di mercato è **gennaio 2018 ($+10,28\%$)**; i due picchi del 2015 (maggio $-4,58\%$, agosto $+8,26\%$) sono invece spiegati onestamente come oscillazioni residue di copertura della fonte nei primi anni (minore densità di `transfers.csv` nel 2012–2014 e assenza delle seconde divisioni in `appearances.csv`).
- **Spese per acquisti e correlazione spuria (`Figure 10–11`):** in 156 mesi si registrano **16.201 acquisti in entrata** (4.950 onerosi) per **56,40 Mld €** totali (**10,61 Mld** in Serie A, 18,8%; picco luglio 2025 con 3,65 Mld).
  - **Sui livelli** la spesa cumulata 12 mesi correla altissimo col valore rose (**$r = 0,89$** su Top-5 e Serie A), ma è **correlazione spuria** dovuta al trend crescente comune!
  - **Sulle differenze prime $\Delta$** la correlazione crolla letteralmente a zero: **$r = 0,00$** sui Top-5 e **$r = 0,03$** sulla Serie A (mentre resta forte la correlazione tra $\Delta$ valore aggregato e $\Delta$ valore medio per giocatore, $r = 0,66$).

---

## 4. Sezione 3 (`capitolo8.tex`): STL, stazionarietà e ACF/PACF (solo Training Set!)

Tutte le diagnostiche di identificazione sono svolte **esclusivamente sul Training Set (primi 120 mesi, giugno 2013 – maggio 2023)** per evitare *data leakage*:

1. **Decomposizione STL robusta (`Figure 12–13`, $s = 12$):**
   - Forza del trend: **$F_T = 0,987$** (in scala log: $0,992$).
   - Forza della stagionalità: **$F_S = 0,115$** ($0,128$ senza aprile 2020; in log: $0,193$).
   - Profilo mensile medio: negativo a **giugno ($-0,30$ Mld)** e **luglio ($-0,46$ Mld)** (scadenze, retrocessioni e ingresso graduale dei vivai neopromossi), positivo da **settembre ($+0,19$)** e **ottobre ($+0,20$)** fino a **febbraio ($+0,18$)**.
2. **Stazionarietà ADF e KPSS (`Tabella 4`, `Figura 14`):**
   - Livelli ($d = 0$, con costante e trend): **non stazionari** (ADF $p = 0,9115$, KPSS $p \le 0,01$).
   - Differenze prime $\Delta Y_t$ ($d = 1$): **stazionarie** (ADF $-10,848, p < 0,0001$; KPSS $0,162, p \ge 0,10$), confermato anche in logaritmi e sull'intera serie. (Nota fine: depurando aprile 2020 il KPSS conferma $d = 1$ con $p \ge 0,10$, mentre l'ADF perde potenza per la coesistenza di stagionalità e cambio di regime 2017–2020; si evita $D = 1$ per non sovra-differenziare su 10 cicli.)
3. **ACF e PACF (`Figura 15`, $N = 119$, soglia $\pm 0,180$):**
   - **Lag 1 nullo** ($\mathrm{ACF}(1) = -0,020$ depurata) $\rightarrow$ base parsimoniosa $p = 0, q = 0$.
   - **Lag 2 moderato** ($+0,236$ depurata), verificato in sensibilità con un $\mathrm{AR}(2)$.
   - **Lag 12 mascherato dal COVID:** sulla serie grezza $\mathrm{ACF}(12) = +0,114$, ma neutralizzando il solo aprile 2020 emerge il picco stagionale significativo **$\mathrm{ACF}(12) = +0,252$** $\rightarrow$ struttura parsimoniosa $\mathrm{SARIMA/SARIMAX}(0, 1, 0) \times (1, 0, 0)_{12}$.

---

## 5. Sezione 4 (`capitolo9.tex`): Gara tra modelli, ablazione pura, diagnostica e Serie A

- **Split cronologico (`Figura 16`):** Training Set = **120 mesi** (giugno 2013 – maggio 2023, 76,9%); Test Set = **36 mesi** (giugno 2023 – maggio 2026, 23,1%).
- **Doppio protocollo:** Holdout unico multi-step a 36 mesi + **Rolling-origin CV** ($h = 1, 6, 12$ mesi su 36, 31 e 25 origini) su tutti i 7 modelli, con test di **Diebold-Mariano** (modelli non annidati) e **Clark-West (2007)** (modelli annidati).
- **Ablazione pura e 100% convergenza MLE:** usiamo lo stesso ordine parsimonioso $(0, 1, 0) \times (1, 0, 0)_{12}$ sia per SARIMA sia per SARIMAX (evitando la mancata convergenza e la cancellazione AR/MA di ordini sovraparametrizzati come $(1,1,1)(1,0,1)_{12}$): **0 mancate convergenze su tutti i fit**.

### Risultati Top-5 (`Tabella 5`, `Figure 17–18`)

| Modello | Holdout 36m RMSE | Holdout MAPE | AIC | BIC | CV $h=1$ | CV $h=6$ | CV $h=12$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline Naive | 2,602 | 7,18% | -- | -- | 0,454 | 1,050 | 1,670 |
| Seasonal Naive ($s=12$) | 3,239 | 9,52% | -- | -- | 1,598 | 1,632 | 1,670 |
| Random Walk + drift | 0,580 | 1,52% | 244,7 | 250,2 | 0,433 | 0,577 | **0,525** |
| Holt-Winters ($s=12$) | 1,225 | 3,03% | -- | -- | **0,394** | 0,764 | 1,162 |
| **$\mathrm{SARIMA}(0,1,0)(1,0,0)_{12}$** | **0,563** | **1,46%** | 245,2 | 253,5 | 0,418 | 0,567 | 0,559 |
| $\mathrm{SARIMAX}$ + step COVID | 1,204 | 3,47% | **114,2** | **125,3** | 0,406 | **0,559** | 0,705 |
| $\mathrm{SARIMAX}$ + COVID + Spesa | 1,209 | 3,49% | 116,2 | 130,1 | 0,407 | 0,560 | 0,707 |

### Le 4 conclusioni chiave da dire all'esame orale
1. **Le spese di calciomercato NON migliorano le previsioni (risultato negativo onesto e rigoroso):**
   - Sui **Top-5**: $\hat{\beta}_{\mathrm{spesa}} = -0,014$ ($p = 0,843$, nullo!), peggiora AIC (`114,2 -> 116,2`) e BIC (`125,3 -> 130,1`), e non migliora l'errore su nessun orizzonte neppure *ex-post* (Clark-West $p = 0,910$ a $h = 1$, $p = 0,874$ a $h = 6$).
   - Sulla **Serie A (`Tabella 7`, `Figura 21`)**: stessa identica conclusione — $\hat{\beta}_{\mathrm{spesa}} = +0,032$ ($p = 0,526$, non significativo), peggiora AIC (`-218,3 -> -216,7`) e BIC (`-207,2 -> -202,8`), e sull'holdout 36 mesi / CV $h=6, 12$ vince il **Random Walk con drift** (`RMSE = 0,093`, `MAPE = 1,39%`), mentre a $h = 1$ i modelli stagionali scendono a `0,092–0,094`.
2. **Il crollo dell'AIC viene unicamente dallo step COVID-19:** isolare il crollo di aprile 2020 ($\hat{\beta}_{\mathrm{COVID}} = -5,82$ Mld, $p < 0,0001$) porta l'AIC da `245,2` a `114,2`, libera la stagionalità reale ($\hat{\Phi}_1 = 0,260, p = 0,0046$), batte significativamente il Random Walk + drift a $h = 1$ (Clark-West $p = 0,0209$) e vince a $h = 6$ (`0,559`).
3. **Perché sul lungo periodo (36 mesi) `SARIMA` batte `SARIMAX + COVID` (`0,563` vs `1,204`)?** Senza la dummy COVID, il crollo $-5,71$ Mld abbassa il drift medio 2013–2023 da `+0,187` a `+0,138 Mld/mese`, che coincide quasi con la crescita più moderata del 2023–2026 (`+0,142 Mld/mese`). Depurando il COVID emerge invece il drift storico più alto (trainato dal boom 2017–2020), che sovrastima il triennio di test.
4. **Diagnostica onesta dei residui (`Tabella 6`, `Figure 19–20`):**
   - Nei modelli **senza dummy COVID** (`SARIMA`), il Ljung-Box grezzo (`p = 0,78–0,98`) è mascherato dal residuo pandemico ($z = -8,44$): togliendolo scende a `0,022` (lag 6) e `0,010` (lag 12).
   - Nei **SARIMAX con drift costante**, Ljung-Box passa al lag 24 (`p = 0,141–0,147`), mentre resta lieve autocorrelazione a breve raggio (`p = 0,012–0,023`) dovuta al cambio di pendenza 2017–2020 vs 2021–2022; **Holt-Winters**, adattando localmente la pendenza, supera Ljung-Box su tutti i lag anche senza aprile 2020 (`p = 0,259 / 0,559 / 0,835`) e vince a $h = 1$ (`0,394`).
   - Tutti i modelli hanno **residui non normali** (Jarque-Bera $p < 10^{-12}$) per la natura a scatti delle revisioni Transfermarkt.

---

## 6. Struttura del codice Python

```text
Parte_2_Time_Series/
├── src/
│   ├── config.py                      # Percorsi portabili (relativi al repo), date split, colori
│   ├── plotting.py                    # Stile 600 DPI e doppio salvataggio figures/ + Relazione/
│   ├── analysis/stl_stationarity.py   # STL, ADF/KPSS (grezzi + verifica COVID/D=1), ACF/PACF
│   └── modeling/sarimax_models.py     # 7 modelli Top-5, Rolling CV, DM + Clark-West, Diagnostica, Serie A
└── scripts/
    ├── 01_etl_transfermarkt.py        # ETL causale + fig01..fig02
    ├── 02_eda_e_shock.py              # EDA, shock, correlazioni livelli vs Δ + fig03..fig11
    ├── 03_run_stl_stationarity.py     # Runner Cap. 8 / Sez. 3 + fig12..fig15
    ├── 04_run_modeling.py             # Runner Cap. 9 / Sez. 4 + fig16..fig21
    └── run_all_pipeline.py            # Esegue l'intera pipeline in un solo comando
```
