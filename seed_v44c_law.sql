-- v44c: Add Speech Pattern Analysis law
INSERT INTO scientific_laws (law_name, definition, scientific_weight, law_domain, failure_modes)
VALUES (
    'Speech Pattern Analysis',
    'Detect emotional and cognitive markers in user syntax. Hesitation signals fuzzy goals. Enthusiasm signals value alignment. Concern signals loss-frame thinking. Source: BioScope Corpus, Prospect Theory.',
    0.95,
    'psychology',
    ARRAY['Missing enthusiasm signals', 'Treating hesitation as rejection', 'Ignoring concern patterns']
)
ON CONFLICT (law_name) DO NOTHING;
