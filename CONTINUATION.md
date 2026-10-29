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
