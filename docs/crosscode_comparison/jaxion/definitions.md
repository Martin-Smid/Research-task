# jaxion vs. náš kód: dynamické tření bodu v ULDM pozadí

jaxion 0.0.13 (JaxionProject/jaxion, commit ef96a94b), JAX 0.6.0 CUDA 12, double precision, P100. Knihovna nebyla upravena;
vlastní skript `jx_friction.py` používá jen veřejné hooky `custom_density`, `custom_kick`, `custom_drift`, `callback`.
Jednotky jaxionu jsou kpc, km/s, Msun (časová jednotka kpc/(km/s) = 0,977792 Gyr); výsledky se přepočítávají na kpc/Gyr.
G = 4,5001e-6 kpc^3/(Msun Gyr^2) (náš kód 4,4985e-6, rozdíl 0,03 %), hbar/m = 7,8428 kpc^2/Gyr, lambda = 0,2614 kpc.
M a rho0 jsou dopočítány z M_Q a z pravidla rho0 M <= 3e12 (M = 3,005e6 / 5,228e7 / 1,743e8 Msun).

## 1a) Přesné definice

### Společné (režimy `point` a `smooth`)
- **Jádro:** Gaussovo v k-prostoru, g(k) = exp(-k^2 eps^2 / 2). V reálném prostoru je to Gaussova hustota
  rho_G(x) = M (2 pi eps^2)^(-3/2) exp(-|x - X|^2 / (2 eps^2)) včetně periodických obrazů, tedy **sigma = eps**
  (s^2 = eps^2, ne 2 eps^2). Stejná konvence jako `gauss = exp(-0.5 k^2 eps^2)` v `sink_friction_scan.py` a v `_smooth_force_grids`.
- **Depozice hmoty:** rho_G se tvoří přímo v k-prostoru, rho_hat = (M/V) N^3 g(k) exp(-i k.X), a vrací se přes `custom_density`
  (jaxion k ní přičte a odečte střední hustotu). X je libovolná poloha bodu, **bez CIC**: Gaussián je vzorkován v buňkách
  přes inverzní FFT na k-mřížce jaxionu (oříznutí na |k_i| <= pi/dx). Pro eps = 0,075 kpc a N = 128 je g na Nyquistu 0,29,
  pro eps = 0,0366 a N = 256 je 0,31 a pro eps = 0,0183 a N = 256 je 0,75 (jádro je za Nyquistem oříznuto).
- **Stejné jádro** se používá pro depozici hmoty do potenciálu i pro sílu na bod u obou přijímačů.
- **Poisson:** spektrální, V_hat = -4 pi G (rho_tot - rho_bar) / k^2, rho_tot = |psi|^2 + rho_G, rho_bar jejich průměr.
- **Gradient:** spektrální, a_hat = -i k V_hat (z celého V včetně psi i vlastního potenciálu bodu).
- **Integrátor:** jaxion kick-drift-kick druhého řádu; bod se kickuje a driftuje stejně jako psi (v = v + a dt/2 ... x += v dt).

### Režim `point` (bodový přijímač)
a(X) = sum_k (-i k V_hat) exp(i k.X') / N^3 (X' = X - dx/2, protože buňky jsou v (j + 1/2) dx):
přesná trigonometrická interpolace spektrálního gradientu v bodě X, **bez CIC a bez vyhlazení**.
Vlastní síla Gaussova bodu je z důvodu symetrie nulová; měřeno bez pozadí (rho0 = 0): změna rychlosti 2e-4 kpc/Gyr za běh
oproti 0,65 kpc/Gyr signálu (0,03 %).

### Režim `smooth` (vyhlazený přijímač)
a = sum_cells (rho_G(x)/M) (-grad V)(x) dV, kde rho_G je stejné vzorkované jádro jako při depozici a grad V je spektrální.
To je gradient pole vyhlazený Gaussem sigma = eps a vyhodnocený v X, (G * F)(X): Gaussova hustota bodu a síla vyhlazená
**stejným Gaussem**, přesně action = reaction na mřížce (hybnost psi + bod se zachovává, viz bilance v tabulce).

### Vestavěná částice jaxionu (`native`)
Depozice hmoty CIC (trilinear), síla = spektrální gradient V, interpolovaný CIC (trilinear) do polohy částice.
Žádný parametr eps ani Plummer; efektivní změkčení je dáno kernelem CIC (rozptyl dx^2/6 na osu při depozici a stejně při
interpolaci), tj. řádově jedna buňka (0,15 kpc při N = 128, 0,075 kpc při N = 256).

### Je můj `smooth` totéž co `smooth_force` v našem kódu?
Z kódu `Sink_N_Body.py` (jen čtení): u nás je smooth_force = sinková hustota je **CIC depozice zkonvolvovaná Gaussem**
exp(-k^2 eps^2/2) (`deposit_to_grid_smoothed`), a síla = **mřížková síla zkonvolvovaná stejným Gaussem a pak trilineárně
interpolovaná** (`_smooth_force_grids` + `_interpolate_force_trilinear`). Můj `smooth` je stejný objekt (Gaussova hustota
a síla vyhlazená stejným Gaussem sigma = eps), **až na kernely CIC**: u vás je účinné jádro Gaussián (x) CIC při depozici
a (x) CIC při interpolaci, u mě čistý Gaussián a přesná spektrální interpolace. Rozptyl CIC je dx^2/6 na osu, takže
sigma_eff^2 = eps^2 + dx^2/6 (depozice); např. eps = 0,0366 kpc při N = 256 (dx = 0,075): sigma_eff = 0,048 kpc, tedy ~30 % širší,
a při eps = 0,0183 kpc 0,0357 kpc (~2x širší). Naše bodové eps je tedy v principu **efektivně měkčí** než moje, a to tím víc, čím je
eps menší oproti buňce. Totéž u bodového přijímače: vy interpolujete trilineárně (účinně vyhlazuje o dx^2/6), já ne.
V `analyze.py` je tahle odchylka záměrně ponechána (porovnáváme implementace, ne ladíme).

## 1b) Počáteční poloha, krok, čas a paměť
- **Poloha:** v souřadnicích jaxionu ([0, 19,2) kpc, periodicky) startuje bod v (4,8; 9,6; 9,6) kpc, tj. v souřadnicích
  boxu [-9,6; 9,6] v (-4,8; 0; 0), rychlost +30 kpc/Gyr ve směru +x, doba 0,32 Gyr (půl boxu) – shodně s naším `sink_friction_scan.py`
  (start v -TRAVEL_FRACTION L / 2). Zadání říká start v (0,0,0); box je periodický a pozadí rovnoměrné, na počátku nezáleží, ale koncový
  bod je tedy +4,8 kpc, ne +9,6 kpc.
- **Časový krok:** dt = 0,5 x limit jaxionu, limit = (m/hbar / 6) dx^2 (= dx^2 / (6 x 7,8428) Gyr). Tj. 2,39e-4 Gyr (N = 128),
  1,06e-4 (192), 5,98e-5 (256), asi 3,8e-5 (320); náš kód: 4e-4, 2,5e-4, 1,4e-4, 9e-5 Gyr (~1,7 až 2,4x větší).
- **Čas a paměť:** tabulka na konci tohoto souboru.
