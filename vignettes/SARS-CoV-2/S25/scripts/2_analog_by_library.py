



# quantinary ammonium compounds:

mkdir product/figures/analog_by_library

obabel raw_data/FDA_quartenary_amines.sdf -O raw_data/FDA_quartenary_amines_clean.sdf -p


python ~/opt/MPLearn/bin/draw_aligned_substances \
       --substances_path intermediate/FDA_quartenary_amines.sdf \
       --substances_id_field=Compound \
       --substances_smiles_field 'Smiles String' \
       --output_path product/figures/analog_by_library/FDA_quartenary_amines.pdf



