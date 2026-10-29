# Pokračování práce — working_with_chat

**CFL v NBodyGas: obě příčiny nalezeny a opraveny (2026-10-16 a 2026-10-26).**
- *Opraveno:* těžká studená koule (1e12 Msun, Rs=5 kpc, h=1e-3, N=128) končila NaN. Příčina: `dt_sub` se počítal jednou na okno, takže skutečné Courantovo číslo během ~80 substepů vyrostlo z 0,5 na 2,95 (vmax 3e3 → 2,4e4), schéma se rozkmitalo a `v = m/rho_floor` přetekla do NaN. Teď se CFL krok přepočítává před každým substepem (`_substeps` v NBodyGas.py); okna s jedním substepem jsou bitově beze změny, cena ~2–6 % jednoho substepu navíc. Při příliš velkém `h` vyhodí `FloatingPointError` s počtem potřebných substepů místo NaN (pro 1e12 a h=1e-3 je to 112 > max_substeps=100). Doporučení pro kolaps: `h = t_ff/100`, t_ff = √(3π/(32Gρ0)) (1e12: 5,85 Myr), `max_substeps` zvýšit.
- *Druhý případ (okno, 2026-10-26) vyřešen:* příčina nebyla CFL ani vakuum samo o sobě, ale nezávislé omezování ρ a hybnosti v MUSCL. Ve vakuové buňce mezi dvěma rozbíhajícími se toky (ρ lokální minimum, hybnost monotónní) vznikl povrch s konečnou hybností a ρ = rho_floor, tedy rychlost ~1e11; Rusanovovy toky ~1e10 daly záporné hustoty a jejich ořez vytvořil hmotu. Oprava (volba B): na plochách, kde je povrchová rychlost větší než 10× (|v|+c) sousedních buněk, se použijí stavy prvního řádu. Referenční sady zůstávají bitově shodné; test `test_gas_vacuum_hole_between_flows.py`, reproducer `resources/diagnostics/cfl_disk_repro.py`.

## Stav k 2026-10-26 (Koios, P100)

**Hotovo**
- Běh na Koios: `KOIOS.md` (ssh-agent, venv, sbatch), reference `baselines_koios_p100/` (bitově reprodukovatelná), srovnání sad `compare_baseline_sets.py`. Quadro vs P100: baryonové případy 1e-14 %, vlnová část ~1e-4 %, sink ~1–3 %.
- Sink: potenciál byl středěný v rohu boxu (ignoroval dolní hranici) a akrece plynu neodebírala energii. Obojí opraveno, testy `test_sink_potential_position.py` a `test_sink_gas_accretion_energy.py`. Akrece se zapíná `sim.sink_accretion = True` (výchozí False). Změnilo se 5 referenčních souborů s sinkem v každé sadě; přegenerována jen sada P100, sady Quadro, RTX 3070 a GTX 1050 Ti jsou pro sink zastaralé.
- Plyn kolem černé díry: 13/13 kontrol (zachování hmoty, energie, rotační křivka = Kepler do 2 %). Ztráta momentu hybnosti 3,6 % za oběh je numerická viskozita Rusanova schématu. **Stabilitu plynu bereme jako hotovou, jak bude** (vnitřní kroužky do 4 kpc ±10 %, vnější 5–8 kpc ~−30 % za Gyr, tam je plyn s cs = 8 pro mělký potenciál příliš horký).
- `main.py`: zjednodušená trpasličí galaxie (soliton 5,3e7 Msun, hvězdy 10 %, plyn 10 %, černá díra 0,4 % hmotnosti solitonu), dvě fáze: relaxace s pomalu zapínanými hvězdami, pak `from_checkpoint` + `start_new_segment` a přidání plynu a černé díry. Čistá relaxace dýchání solitonu (±13 %, perioda ~0,75 Gyr) neutlumí; pomalé zapínání ano (rozkmit 0,4 %). Po přidání plynu naráz se dýchání vrací s poloviční amplitudou (13 %), což bereme jako přijatelné.
- Energie: skok při t=0 zmizel, drift +0,28 % za Gyr. Virial s izolovaným W: soliton sám 1,004, celek −0,2 % z |W| na začátku.
- Obrázky z `plot_energies.py`, `plot_density.py`, `Plot_max_vals.py`, `plot_from_snapshots.py` se ukládají do složky simulace.

**Co dál**
1. Konvergence: hotová pro N = 64/128/256 s `complex128` (viz Doplněk); zbývá numerická viskozita plynu (RK2 v čase pro MUSCL?).
2. Volitelně zapínat plyn postupně (jako hvězdy), případně relaxovat rovnou s plynem.
3. ~~Dohledat únik hmotnosti vlnové funkce (0,04–0,09 % za Gyr)~~ vyřešeno: byl to `complex64`, viz Doplněk 2026-10-26.
4. Částice a sinky pořád používají zamrzlý potenciál v čase; podezřelý potenciál samointerakce (viz CLAUDE.md, body 2 a 3).
5. Bondiho akrece používá konstantní `gas.cs`; rychlost akrece bereme jako numerický test, ne fyzikální predikci.
6. Zkontrolovat starší simulace s černou dírou v boxech, které nezačínají v 0.
7. Přegenerovat reference na ostatních GPU; commit (zatím nic nebylo commitnuto).
8. Úklid balíčkování a dokumentace, licence a citace (viz seznam níže).

## Doplněk 2026-10-26 (večer)

- **Přesnost ψ:** vlnová funkce byla v `complex64`, takže každý krok zaokrouhloval na float32 a norma ψ klesala úměrně počtu kroků (−0,3 % za Gyr při h = 1,25e-4; drift energie 0,28 % → 1,2 % při 4× menším kroku, nezávisle na N). Opraveno na `complex128` (2 řádky, `Wave_function_class.py`, `Wave_Packet_Class.py`). Změna výsledků ~1e-7, `tests/run_all.py` prochází, sada `baselines_koios_p100` přegenerována (bitově reprodukovatelná). Sady Quadro/RTX/GTX jsou zastaralé pro všechny případy.
- **Konvergence (fáze 2, 1 Gyr, N = 64/128/256):** s `complex128` zmizel drift energie (−0,036 / −0,024 / −0,015 % místo +0,25 / +0,28 / +1,16 %) i ztráta hmotnosti ψ (~1e-10 %); `complex128` nestojí měřitelný čas (N = 256: 2 h 46 vs 2 h 42). Soliton konverguje (ρ_max 1,500 / 1,508 / 1,511e7; profil pro r ≥ 1 kpc mezi 128 a 256 do 1 %, pod ~0,5 kpc pomaleji), dýchání ρ_max (10,3 / 12,2 / 13,4 %) je fyzikální (soliton kmitá po zapnutí baryonů); jeho růst s N je hlavně lépe rozlišený vrchol (max 1,662 / 1,704 / 1,727e7, přírůstky 0,042 / 0,023), ne jiná fyzika. Plyn: ztráta momentu hybnosti −30,9 / −18,2 / −9,9 % (klesá pomalu, zdánlivý řád ~0,6 ze tří bodů), vφ v 1–3 kpc 5,28 / 6,34 / 6,55, vnější plyn 3–6 kpc nekonverguje. N = 32 spadne ve fázi 2 na NaN plynu (příliš hrubé). Zbývající nekonvergovaná část je numerická viskozita plynu (jedna Eulerova fáze MUSCL).
- **Částice a sinky:** nové testy `test_particle_kepler_orbit.py` (integrátor je učebnicové DKD, skutečný krok určuje kritérium podkroků), `test_sink_force_vs_analytic.py` (pole sinku sedí na analytiku, použitelné do r ≤ L/4, dál anizotropie periodické mříže), `test_particle_breathing_soliton.py` (proměnný zdroj: první řád v h kvůli zamrzlému potenciálu, chyba ~1e-3 r0, přijato). Kruhová dráha se Keplerovou rychlostí `sqrt(GM/r)` v periodickém boxu není kruhová (síla je při r = L/4 o ~6,5 % slabší); použít r0 ≲ L/10 nebo rychlost z mřížky. Minimální skript: `testing_baryons.py`.

## Handoff 2026-10-27

Další fáze: kolize černých děr v solitonu (bez plynu). **Blokátor: sinky se navzájem nepřitahují** (dva sinky 4 kpc od sebe zůstanou po 0,1 Gyr přesně 4,0000 kpc; sink se hýbe jen v `forces_environment`). Plán a stav repa/dat: viz sekce „Handoff 2026-10-27“ na konci `CLAUDE.md`.

## Cíl a pravidla

Zjednodušit simulátor, omezit zbytečné výpočty a závislosti a připravit dokumentaci pro publikování. Zachovat strukturu autorova kódu a malé, čitelné změny. Při každé úpravě numerického výpočtu porovnat mřížky, částice a energie s původními referencemi; změna výsledků znamená zastavit a vyšetřit příčinu. Referenční data nepřepisovat jako způsob opravy testu.

`working_with_chat` obsahuje testy, reference a historii v resources/diagnostics; main má stejný produkční kód a příklady bez tohoto vývojového adresáře a chats_playground.py. Tento soubor patří pouze do pracovní větve. Na novém počítači začít touto větví, přečíst resources/diagnostics/README.md a docs/rotation_curves.md, připravit CUDA/CuPy prostředí a ověřit testy před změnami.

## Co už je hotové

- Gravitace: stejný potenciál pro wave kick a baryon kick se znovu použije pouze v rámci stejného integračního stavu. Konstantní jmenovatel Poissonova výpočtu se připraví jednou v Propagator; pořadí kroků a aktualizace hustoty se zachovaly. Autor naměřil přibližně 40–45 -> 50–55 FPS.
- Diagnostics_Class.py: Evolution předává aktuální stav a poskytovatel sink potenciálu, Diagnostics počítá energie/profily a organizuje rotace; výsledky jdou přes Evolution do Scribe. Vnořená smyčka, která počítala vnitřní energii plynu jednou za každý baryonový systém (s plynem a sinkem 2×), byla opravena: U_iso plynu se počítá jednou. Mění to jen logované energie, ne vývoj.
- Rotation_Curve_Class.py: společný GPU výpočet pro plyn a částice, periodický hmotnostní střed, odečtení pohybu těžiště a osa z celkového momentu hybnosti. Výchozí cutoff je válcový poloměr obsahující 50 % hmotnosti, ne 50 % maximální hustoty; neurčená osa vrací stav místo smyšlené křivky. NBody/NBodyGas volají společnou třídu, Diagnostics vrací malé profily a metadata do Evolution, Scribe zapisuje rotational_velocity.dat a rotation_frames.csv.
- plot_rot_curves.py: bez argumentů poslední běh a oba typy komponent; -g/-N filtr, více adresářů překryje více běhů, --no-show pouze uloží. examples/plot_rotation_curves.py sdílí implementaci.
- Vazba plynu na gravitaci (NBodyGas.py, Evolution_Class.py, 2026-10-16): (1) energie z gravitace se plynu dává jako tok hmoty přes stěnu buňky × rozdíl potenciálu (`_sweep_axis`, `phi_total`), takže součet přes buňky přesně odpovídá změně W; (2) kick–hydro–kick s potenciálem obnoveným po hydru (`potential_update`, jedno Poissonovo řešení navíc na okno, +3–6 % času okna); (3) pojistka vnitřní energie `_add_gravity_energy_correction` drží E−K nad floorem, dodanou energii sčítá `E_guard_injected`. Evolution plyn zásobuje přes `_set_gas_potential`. Bez `phi_total`/`potential_update` zůstává původní chování (bitově).
- Energie v main.py konfiguraci (N=64, t=0,06 Gyr): drift +3,6 |E0| → +0,1 %; N=128: −1,5 %. Pojistka dodala ~1,5 % |E0|. Výsledky plynu se změnily: hustota/rychlosti o ~1e-3 % v regresním případu, vnitřní energie studeného plynu v nerozlišeném potenciálu o desítky %. Starý kód nekonvergoval s rozlišením (hmota do 1 kpc, N=64→160), nový ano. Ramena („kříž") v řezech main.py vznikají na hranicích boxu z neperiodické počáteční podmínky (rotace a hustota se na stěnách nespojují), jsou i ve starém kódu.
- NBodyGas.py dále: (4) `_reset_emptied_cells` po každém směrovém sweepu: buňka vyprázdněná pod rho_floor dostane nulovou hybnost a energii na floor místo `v = m/rho_floor` (bitově beze změny na benchmarcích, drift energie disku stejný); (5) adaptivní `_substeps` (viz CFL výše).
- Validace volného pádu studené koule (N=128, 1e12, h=t_ff/100, bez solitonu a sinku): r50/r50₀ = 0,85 / 0,57 / 0,20 při t/t_ff = 0,5 / 0,8 / 1,0, analytika 0,84 / ~0,52 / 0; zaostává o ~2 % do 0,5 t_ff a ~9 % u 0,8 t_ff, kolaps ~1,05 t_ff, špička ρ_max ≈ 150–265 ρ0 (omezeno rozlišením). Je trvalým testem (`tests/test_gas_free_fall_collapse.py`, N=64, tolerance 2/4/8 % při 0,25/0,5/0,75 t_ff). Funkce `free_fall_estimate(mass, radius)` v NBodyGas.py vypíše t_ff, doporučené `h = t_ff/100` a `total_time = 1,3 t_ff`.
- tests/ už není v .gitignore. Nové testy: práce gravitace vs. změna W, kladná vnitřní energie, U_iso jednou, Sodova trubice, Jeansova nestabilita (obě proti analytice, tolerance v souborech), vyprázdněná buňka bez hybnosti, adaptivní substepy. Pouštět `tests/run_all.py` (~1 min).
- Úklid: odstraněno 188 simulací před 2026-10-04 (1,86 GiB); dnešní běhy a reference zachovány. Jednorázové migrační skripty jsou v diagnostickém archivu a nelze je přímo znovu spustit.

## Ověření a omezení

Původní čtyři případy měly bitově shodné mřížky a energie; rozšířené porovnání optimalizace zahrnovalo 12 případů. Oddělení diagnostik a rotace ověřeno v sedmi případech s 49 energetickými vzorky. Restart bitově shodný; drobné segmentové rozdíly existovaly už ve starém kódu. Osm analytických testů rotací prošlo i po úklidu. Povolené změny uložených výstupů po nové rotaci jsou pouze rotational_velocity.dat a rotation_frames.csv.

Neuzavřeno po změnách plynu: (a) referenční sady pro GTX 1050 Ti (`baselines/`) a RTX 3070 (`baselines_rtx3070/`) mají plynové případy z kódu před změnami (nový kód se jim záměrně neshoduje); sada pro Quadro P4000 (`baselines_quadro_p4000/`) byla 2026-10-16 přegenerována (plynové případy z kódu po změnách, ostatní bitově beze změny), `compare`, `diagnostics-check` i optimalizační kontrola prochází bitově; (b) tepelná energie studeného plynu v nerozlišeném potenciálu je slabé místo (rozdíl práce přes stěny a ve středu buňky padá do E−K); (c) zbytek driftu energie při N=128 (−1,5 %) není vysvětlen; (d) chybí validace volného pádu koule, orders 4/6 pro plyn zůstávají odmítnuté.

NBodyGas podporuje řád evoluce 2; řády 4/6 mají záporné podkroky a jsou odmítnuty. Plynový dt musí být konečný a nezáporný. Nezaměňovat tato omezení s dosud nevyšetřeným problémem CFL. Nakloněný disk N=128 se správnou osou není důkazem fyzikální rovnováhy.

```powershell
python chats_playground.py --help
python chats_playground.py compatibility-check
python resources/diagnostics/rotation_playground.py
python resources/diagnostics/check_optimization.py --help
```

Pro porovnání rotací použít podporovaný přepínač --allow-rotation-changes podle nápovědy playgroundu. Lokální .venv není přenosná a data simulací jsou ignorovaná; uložené referenční NPZ a manifesty v pracovní větvi jsou přenosné. Na Windows může být nutné použít .venv/Scripts/python.exe místo systémového aliasu python.

## Další práce po CFL

1. **Reduce structural clutter.** Consolidate the two system_fucntions.py modules after checking their callers, replace wildcard imports with explicit imports, and move plotting helpers out of engine dependencies. Extract the repeated baryon advancement code while preserving the integration sequences.
2. **Make installation reproducible.** Add package metadata, distinguish runtime dependencies from plotting/development dependencies, and verify a clean installation. One concrete gap: the engine imports tqdm, but requirements.txt omits it. Tracked editor files and historical outputs also need a deliberate cleanup.
3. **Document the verified behavior.** Replace the README's development diary with installation, a minimal runnable example, units, boundary conditions, supported components, and restart instructions. Add numerical-method documentation, checkpoint-format documentation, and reproducible validation cases with stated tolerances. Before release, settle licensing and citation information with the author.

Postupovat po malých změnách, nejprve měřit a porovnávat výsledky. Vývojové experimenty držet v working_with_chat; do main přenášet ověřený kód, příklady a uživatelskou dokumentaci.

## Pravidlo pro commity

Commity jsou pouze na jméno autora (Martin-Smid). Do commit zpráv ani popisů PR nepřidávat `Co-Authored-By` ani jiné řádky s Claude/Anthropic, aby GitHub nezobrazoval Clauda jako přispěvatele. Platí i pro Claude Code běžící na Koios.


## Dva články a nástroje pro soliton s černou dírou (2026-10-29)

### Nástroje (hotové, necommitnuté; testy v `tests/test_dressed_soliton.py`, mimo `run_all`, ~5 min na GPU)
- `DressedSoliton.py` (1D, CPU/scipy): radiální základní stav SP s centrální černou dírou, Gaussovskou (šířka `eps` = `softening_bh` běhu) nebo bodovou. Shooting v jednom vlastním čísle.
  `python DressedSoliton.py --selftest` (volný soliton = `GroundState(1).dat` na 7 míst, vodíkový limit), `python DressedSoliton.py --m 1e-21 --M_sol 5.57e7 --M_bh 4e6 --eps 0.005 --out profil.dat`
  zapíše profil (`r phi U`) pro `Wave_vector_class` (`packet_type=profil.dat`, `desired_soliton_mass` = vypsaná hmotnost, faktor ≠ 1 platí jen bez BH). Z Python: `solve_physical(m_ev, M_bh, M_sol, eps)`.
- `DressedSoliton3D.py` (3D, GPU): imaginary-time Strang na stejné periodické mřížce jako simulace (stejná Poissonova rovnice se zrušenou střední hustotou, analytický Gaussovský potenciál sinku),
  hmotnost se renormalizuje na `m_sol`. `relax(sim, m_sol, m_bh, bh_position, eps, pin_centroid=None)`, `save(...)`, `load_into(wave_function, soubor.npz)` po `initialize_simulation()`.
  Časový krok se zmenšuje po úrovních (pevný bod rozděleného kroku je vlastní stav lehce změněného H, reziduum klesá ~tau^1,3-2); cíl rezidua 3e-4. CLI: `python DressedSoliton3D.py --N 192 --half_box 1.0 --M_bh 4e6 --eps 0.005 --out dressed.npz`.
  `pin_centroid`: bez něj se stav s BH mimo střed usadí na BH; s ním je to soliton držený v bodě a zdeformovaný slapem BH (vázaný stav, ne stacionární).
- Ověření: 3D (BH ve středu, N=128, box 1,5 kpc) vs 1D: centrální hustota 1,1575e11 vs 1,1584e11 (0,08 %), max rozdíl profilu 8e-4 ρ_c, reziduum 1e-4; vodíkový limit hmotnosti přesně (1e-8).
- Neplatí: profily z `ModosFinder.py` / `ModosDbscan*.ipynb` mají φ(0)=1 (pevná centrální hustota), přeškálování hmotností je u profilu s BH neplatné, hledání je náhodné.

### Článek A: otestovaný kód pro ostatní (software + validace)
Cíl: SP s Fourierovým split-step, baryony (částice, plyn) a sink částice; srovnatelné s PyUltraLight, s testy a návodem. Možná místa: JOSS (software), Astronomy & Computing, RNAAS (krátké).
Co už máme: regrese bitově na P100 (`baselines_koios_p100`), testy (Kepler, binárka sinků, slučování, `smooth_force`, energie sink-sink, Sod, Jeans, volný pád...), exaktní lineární benchmark v boxu
(`tests/sink_friction.py`), mapa C(M_Q, ε/λ) a konvergence (`tests/sink_friction_scan.py`, `t2_aggregate.py`), complex128, řešiče solitonu 1D/3D.
Co chybí (pořadí): (1) README se skutečným návodem (instalace, minimální příklad, jednotky, okrajové podmínky, restart), `requirements.txt` v UTF-8 a bez zbytečných balíků, licence, CITATION, `.idea` pryč, sloučit `system_fucntions.py`;
(2) výchozí hodnota `smooth_force` (rozhodnutí; zatím False), popis všech parametrů sinku a doporučené rozsahy (ε/λ ≲ 0,15, λ ≥ 3 buňky); (3) výkon a paměť (P100: N=384 nejde, ~5 min / případ N=256) tabulka a škálování;
(4) srovnání s nezávislým kódem na stejném testu (PyUltraLight, po ověření, že umí uniformní pozadí) a na pár bodech literatury (T5); (5) jednotná mapa platnosti (viz B); (6) regenerace referenčních sad pro další GPU (Quadro/RTX/GTX jsou po complex128 zastaralé);
(7) dokument „known limitations": periodický box (zbytková gravitace obrazů, střední hustota -(4π/3)G m r/V platí do r ≈ L/4), sponge je complex64, bodový/Gaussovský BH, gas nepatří do jádra článku.
Poctivě: článek A je software + validace, ne nový fyzikální výsledek.

### Článek B: dynamické tření sinku v homogenním a solitonovém FDM pozadí (rozšíření 2403.09038)
Na čem stojí (už máme, ověřeno, čísla v `plans.md` sekce 7): (i) exaktní lineární teorie pro konečný periodický box s Gaussovským sinkem (diskrétní součet módů) souhlasí se simulací na 0,5-1 % pro M_Q ≥ 10;
(ii) limita ε → 0 souhlasí s Lancasterem 4.21 (C 3,276 vs 3,268 při M_Q 17,4; M_Q = 1: +2 %; M_Q = 0,3: +7-9 %, ještě konverguje); (iii) bodový přijímač porušuje hybnost a energii úměrně (ε/λ)², `smooth_force` ji obnoví (~N^-2);
(iv) konvergence v N a H pod 0,5 %; oscilace při M_Q ≤ 1 jsou fyzikální; (v) čtení 2403.09038: jejich setup má λ ≈ 2,3 buňky a ε/λ ≈ 0,2 při špičkové rychlosti, což je v oblasti našich několika procent.
Tvrzení, která by článek mohl nést (každé jen po splnění uvedených kroků): (a) mapa C a chyb hybnosti/energie v (M_Q, ε/λ, N) s pravidlem platnosti; (b) exaktní benchmark jako veřejný test pro ostatní kódy; (c) odhad systematiky publikovaných t_stop a dalších veličin
vůči ε/dx, N, boxu a okrajům (sponge vs periodické); (d) chování BH v solitonu: čas zastavení, reheating, poloha vůči těžišti solitonu, stacionarita dressed stavu, adiabatický růst, kick. Novost nelze tvrdit před přečtením celých validačních částí Boey+2024, Wang & Easther 2021,
Boey+2025 (2504.16348), Buehler & Desjacques (2207.13740), Lancaster+2020; z druhé ruky: 2504.16348 a podobné.
Kroky (nejlevnější a nejvíc rozhodující první):
1. Dokončit T2: N=320 pro M_Q 1 a 0,3 (běží), případně menší box při N=256 pro ε/λ < 0,07; rozhodnout o výchozí `smooth_force`. [hodiny]
2. Test A (stacionarita): `DressedSoliton3D` + BH ve středu, pustit real-time, měřit drift hustoty/polohy BH vs ε/dx, M_BH/M_sol, N. [minuty až desítky minut] Tím se ověří i `Sink_N_Body` v solitonu.
3. T5: přečíst 2403.09038 stránky 12-13, postavit jejich setup (soliton 5,57e7 z Eq. 4, BH 4e6 v klidu na (0,125, 0,125, 0), box 2 kpc, N = 192, ε = 0,5 dx), t_stop a reheating; ověřit proti A/M + B (A = 1,38e8 Msun Myr, B = 12,4 Myr). [hodiny na běh]
4. T6: citlivost na ε/dx ∈ {0,25, 0,5, 1, 1,5}, N ∈ {128, 192, 256}, bodový přijímač vs `smooth_force`, zmrazený soliton (kontrola), sponge vs periodické okraje (`use_sponge`, V0 = 0,6, zkontrolovat jednotky a complex64). [desítky běhů, 1-2 dny GPU; prioritizovat]
5. Adiabatický růst (dM = (M_f - M_i) / N_steps, doba růstu ≫ dynamická doba solitonu ~ Myr; ověřit konvergenci s dobou růstu), porovnání s dressed stavem z ODE; potom kick (rychlost BH ve středu) a měřit tlumení vs M_Q. [hodiny]
6. Mapa M_Q v solitonu: M_Q,eff ≈ M_sol/M_BH × O(1) (z profilu spočítat přesně), kdy přestává lineární režim platit; srovnat s uniformním pozadím.
7. Nezávislé ověření (druhý kód) aspoň na jednom bodě uniformního testu a na T5; bez něj zůstane „kód reprodukuje sebe".
8. Teprve potom text: čísla z `plans.md`, grafy `resources/data/sink_friction_t2_summary`, tabulky, veřejné skripty.
Rizika: mřížkové artefakty diagnostik (viz energie sinku: analytický self-term), `smooth_force` při ε < 0,5 buňky (hybnost +3,9 % při M_Q = 0,3, příčina neověřena), N = 384 nejde na P100, jedna volná GPU, diplomka neslouží jako zdroj čísel.

## Handover 2026-10-29 (večer) — kde jsme, co dál

**Stav repa (`working_with_chat`, Koios).** Commity `9c2ad17` (smooth_force), `b1de09e` (párová síla s² = ε_a²+ε_b², analytická energie sink-sink v Diagnostics, test A), `c135f14` (`max_values.csv` průběžně s energií, odstraněn starý check `Deviation`/`mass diff`, nahrazen hlášením hmotností). Nepushnuto je 9 commitů (6 starých + 3 z dneška); push dělá uživatel přes VS Code.
Necommitnuté (moje): `DressedSoliton.py`, `DressedSoliton3D.py`, `tests/test_dressed_soliton.py`, `tests/README.md`, tato sekce, `tests/sink_friction_scan.py` (řízení přes env proměnné, T2), `tests/t2_aggregate.py`, `slurm/koios_t2.sbatch`, `slurm/koios_dressed_test.sbatch`, `correct_params.py` (nový). Uživatelovo (nesahat): `testing.py`, `plot_from_snapshots.py`, `latest_comparison.json`.
Joby: `3649545` (cpu_int, zaseklý pending; zrušit jen se souhlasem), `3649691` (`bh320_check`: diagnostika runawaye, viz níže; výsledek v `slurm/bh320_check-3649691.out`, běhy v `resources/data/simulation_20261029_21*`; skripty `~/conv/bh320_smooth.py`, `~/conv/bh320_eps125.py`, profil `resources/solitons/Dressed_..._eps0.0125.dat` jsou moje, smazat jen po souhlasu).

**Nové od posledního zápisu (podrobnosti a čísla v `plans.md`, sekce 8).**
- Nástroje na soliton s BH: 1D `DressedSoliton.py` (CLI ukládá do `resources/solitons/`, G z astropy jako Simulation_Class; ModosFinder má G o 3e-5 vyšší), 3D `DressedSoliton3D.py`, `correct_params.py` (z N, M_sol, m, M_bh vyplní box, ε, H, čas, kontroly). Ověřená ruční simulace (stacionární běh i kick) je v `plans.md`.
- Běhy uživatele (`testing.py`): N=256 box ±2 s H = 1,5e-4 je 5,7× nad limitem (kód jen varuje); N=256 s ε = 0,96 buňky pulzuje 1,0-1,29× ρ0; **N=320 s ε = 0,36 buňky a r_B = 1,4 buňky: exponenciální runaway** (hybnost ψ+BH roste 10× za 1,7 Myr, BH uteče po úhlopříčce, E +17 %). Hypotéza: nereciproční bodový přijímač (`smooth_force=False`) při ε < 1 buňka; alternativa: mřížková anizotropie / Nyquist. Neověřeno, kontrolní běhy `3649691` (A: smooth_force=True, B: ε = 1 buňka).

**Co dělat dál (pořadí).**
1. Vyhodnotit `3649691` (BH poloha, ρ_max/ρ0, E_total, hybnost ψ+BH ze snapshotů `wf_0`), zapsat do `plans.md`. Z toho rozhodnout o výchozí hodnotě `smooth_force` (důkazy: T2/T3 hybnost a energie, test A, runaway).
2. Ověřit doporučení `correct_params.py` na případu uživatele (N=320, M_sol 1e8, M_bh 5e7: box ±0,389, ε = 0,00243, H = 5,8e-7, 7400 kroků) jako stacionární kontrolu; tím se potvrdí/vyvrátí pravidla (r_B ≥ 4 buňky, ε = 1 buňka, fáze potenciálu < 0,1 je jen odhad).
3. Zvážit, aby `Simulation_Class` při H nad limitem stability skončila chybou (nebo to šlo vypnout); teď jen varuje a běh je tichý nesmysl.
4. Vyhodnotit N=320, M_Q=0,3 (`slurm/t2n320q03_*`, joby 3649676-79; zatím nikde nezapsáno) a doplnit tabulku ε→0.
5. T5: přečíst `2403.09038` str. 12-13, postavit setup (soliton 5,57e7, BH 4e6 v klidu na (0,125, 0,125, 0)), použít `correct_params.py` pro box/ε/H (u 192³ vychází box ±0,81, ε = 8,4 pc, H = 6,9e-6), t_stop a reheating; pak T6 (ε/dx, N, smooth_force, sponge vs periodické okraje, zmrazený soliton).
6. Adiabatický růst BH (dM = (M_f−M_i)/N_steps), kick, 3D `pin_centroid` v simulaci (zatím jen v testu mimo simulaci).
7. Sponge: `use_sponge`, `sponge_V0` = 0,6 (jednotky nejasné), pole je complex64; ověřit před použitím.
8. Commitnout nástroje (po souhlasu), uklidit; kód-článek (A) a článek B podle sekce výše.

**Pasti, které jsme potkali (aby se neopakovaly).** `wave_function.psi` je do `evolve()` numpy; loader profilů čte float32, interpoluje lineárně a mimo tabulku dává 0 (proto první řádek profilu r = 0); `--export=…MQ_LIST=1,0.3` v sbatch rozdělí hodnotu čárkou; `/tmp` je lokální na uzlu; N=384 přeteče 16 GB P100; PowerShell kazí uvozovky v `ssh … "…"` (skripty psát do souboru a `scp`); jedna volná GPU, joby čekají za sebou; diplomka obsahuje chyby, jen inspirace.
**Aktualizace (noc 2026-10-29).** Kontrolní běhy `3649691` (zrušeny uživatelem v 20,3 Myr běhu B) vyvrátily hypotézu o nereciprocitě: runaway hybnosti nastává i se `smooth_force=True` (jen o ~5 Myr později) a i s ε = 1 buňka (B). Nejlépe podporovaná hypotéza: kusp s r_B = 1,4 buňky není rozlišený a schéma pak nezachovává hybnost interakce ψ-BH (netestováno). Nejdůležitější další krok: spustit doporučený případ z `correct_params.py` (N=320, box ±0,389, ε=0,00243, H=5,8e-7). Podrobnosti v `plans.md` sekce 8. Test `tests/test_dressed_soliton.py` je obnovený z mé kopie (po obnovení znovu nespuštěn).