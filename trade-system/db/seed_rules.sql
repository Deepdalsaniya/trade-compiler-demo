-- Seed a few demo rules so the engine works on day 1

INSERT INTO rules (name, priority, active, condition_json, add_forms)
VALUES
-- Base docs for US -> DE/EU route
('US->DE base', 10, TRUE,
 '{"origin":"US","destination_in":["DE","EU"]}',
 '["CI","PL","SLI","EEI_Worksheet","EU_SAD","Generic_CoO"]'),

-- If HS chapter/prefix indicates electronics (85), add EU docs
('HS 85 electronics', 20, TRUE,
 '{"hs_prefix_in":["85"]}',
 '["CE_DoC","RoHS_Declaration","REACH_Declaration"]'),

-- US EEI threshold
('Value > 2500 EEI', 30, TRUE,
 '{"value_gt":2500}',
 '["EEI_Worksheet"]'),

-- Wood packaging triggers ISPM-15 statement
('Wood packaging', 40, TRUE,
 '{"packaging_eq":"wood"}',
 '["ISPM15_Statement"]');

