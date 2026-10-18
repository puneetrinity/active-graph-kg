import {readFileSync,readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
const definitions=new Map();
for(const file of readdirSync('db/migrations').sort().filter(file=>/^0\d\d_.*\.sql$/.test(file)&&file<'036_')) {
  const sql=readFileSync(`db/migrations/${file}`,'utf8');
  for(const match of sql.matchAll(/CREATE(?: OR REPLACE)? FUNCTION public\.(\w+)\([\s\S]*?\bAS (\$[a-zA-Z_0-9]*\$)[\s\S]*?\2;/g))definitions.set(match[1],match[0]);
}
const signatures={candidate_index_claim:'text,integer,jsonb,uuid,uuid,bigint,text',
  candidate_index_complete_extract:'uuid,uuid,bigint,text,text,jsonb,jsonb,double precision,text,jsonb,text,text',
  candidate_index_fail:'uuid,uuid,bigint,text,integer',
  candidate_index_plan_recovery:'text,jsonb,text',
  candidate_index_read_sourcing:'vector,text,text,integer,integer'};
const quote=value=>"'"+value.replaceAll("'","''")+"'";
const output=[];
for(const [name,signature] of Object.entries(signatures)) {
  const original=definitions.get(name);if(!original)throw Error(name);
  const marker=/\bAS (\$[a-zA-Z_0-9]*\$)/.exec(original);
  const body=original.slice(marker.index+marker[0].length,original.lastIndexOf(marker[1]));
  const replacements=[];
  const add=(from,to)=>{const count=original.split(from).length-1;if(!count)throw Error(`missing ${name}: ${from}`);replacements.push({from,to,count});};
  if(name!=='candidate_index_read_sourcing')add('^candidate-index-extraction:v2:[0-9a-f]{64}$','^candidate-index-extraction:v[23]:[0-9a-f]{64}$');
  if(name==='candidate_index_complete_extract')add('    FOR evidence IN SELECT value FROM jsonb_array_elements(p_evidence) LOOP',
    `    IF g.policy->>'extraction_adapter_version' ~ '^candidate-index-extraction:v3:[0-9a-f]{64}$'
      AND public.candidate_index_history_valid(p_namespace->'employment_history',input_text) IS DISTINCT FROM true THEN
      RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_extraction_incomplete'; END IF;
    FOR evidence IN SELECT value FROM jsonb_array_elements(p_evidence) LOOP`);
  if(name==='candidate_index_read_sourcing') {
    add('c.global_candidate_id linked_global_id,e.namespace,','c.global_candidate_id linked_global_id,e.namespace,g.policy,');
    add("'currentTitle',item.namespace->'current_title'",`'employment',CASE WHEN item.policy->>'extraction_adapter_version' ~ '^candidate-index-extraction:v3:[0-9a-f]{64}$'
          THEN (SELECT coalesce(jsonb_agg(jsonb_strip_nulls(jsonb_build_object('title',r->'title',
            'employmentType',r->'employment_type','start',r->'start','end',r->'end','ongoing',r->'ongoing'))),'[]'::jsonb)
            FROM jsonb_array_elements(item.namespace->'employment_history') r) ELSE '[]'::jsonb END,
        'currentTitle',item.namespace->'current_title'`);
  }
  output.push(`DO $forward$ DECLARE definition text; source text; BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc INTO STRICT definition,source FROM pg_proc p
  WHERE p.oid='public.${name}(${signature})'::regprocedure;
 IF encode(sha256(convert_to(source,'UTF8')),'hex')<>${quote(createHash('sha256').update(body).digest('hex'))} THEN
  RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 ${replacements.map(({from,to,count})=>`IF (length(definition)-length(replace(definition,${quote(from)},'')))/length(${quote(from)})<>${count} THEN RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 definition:=replace(definition,${quote(from)},${quote(to)});`).join('\n ')}
 EXECUTE definition;
END $forward$;`);
}
console.log(output.join('\n\n'));
