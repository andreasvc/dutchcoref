declare namespace saxon="http://saxon.sf.net/";
declare namespace ud="https://universaldependencies.github.io/docs/" ;

declare option saxon:output "omit-xml-declaration=yes";
declare option saxon:output "indent=yes";

declare  variable $DIR external ;
declare  variable $MODE external ;
declare  variable $ENHANCED external ;

(: should work in coordinations like te laten reizen en te laten beleven,
   and recursive cases: Andras blijft ontkennen sexuele relaties met Timea te hebben gehad ,
	          .. of hij ook voor hen wilde komen tekenen :)
declare function local:xcomp-control($node as node(), $so_index as xs:string) as node()* {
	for $xcomp in 
	    $node/ancestor::node//node[(@rel="vc" or (@cat="inf" and @rel="body")) (: covers inf ti oti :)
		                           and node[@rel=("hd","predc") and @ud:Relation="xcomp"]  (: vrouwen moeten vertegenwoordigd zijn :)
		                           and node[@rel="su" and @index]/@index = $so_index
		                          ]
 	return
	<headdep head="{local:internal_head_position($xcomp)}" dep="nsubj"/>                            
};

(: alpino NF specific case, controllers with extraposed content are realized downstairs :)
declare function local:upstairs-control($node as node(), $so_index as xs:string) as node()* {
    for $upstairs in 
		$node/ancestor::node[node[@rel="hd" and @ud:pos="VERB"] 
				                  and node[@rel="vc"]
				                  and node[@rel=("su","obj1","obj2") and not(@pt or @cat)]/@index = $so_index
				                 ]      
	return
	<headdep head="{local:internal_head_position($upstairs)}" dep="nsubj"/>                            
};

(: een koers waarin de Alsemberg moet worden beklommen :)
declare function local:passive-vp-control($node as node(), $so_index as xs:string) as node()* {
    for $passive_vp in 
        $node/ancestor::node//node[@rel="vc" and @cat="ppart" 
	                               and node[@rel="hd" and @ud:Relation="xcomp"] 
	                               and node[@rel="obj1" and @index]/@index = $so_index ]	
	return
	<headdep head="{local:internal_head_position($passive_vp)}" dep="nsubj:pass"/>                            
};

declare function local:anaphoric_relpronoun($node as node()) as node()* {	
	(: works voor waar, and last() picks waar in 'daar waar' cases :)
	(: dont add anything for hij werd voorzitter, wat hij nog steeds is (otherwise self-reference) :)
	(: for loop ensures correct result if N has 2 acl:relcl dependents :)
	for $anaphoric_relpronoun in
			$node/ancestor::node[@cat="np" and local:internal_head_position(.) = $node/@end]/
			       node[@rel="mod"]/node[@rel="rhd"]/descendant-or-self::node[@pt="vnw" and not(@ud:HeadPosition = $node/@end)][last()]
	return 
	<headdep head="{$anaphoric_relpronoun/@ud:HeadPosition}" dep="{$anaphoric_relpronoun/@ud:Relation}" />

}; 

(: Glastra en Terlouw verzonnen een list --> nsubj(verzonnen,Glastra) nsubj(verzonnen,Terlouw) :)
declare function local:distribute_conjuncts($node as node()) as node()* {
	let $coord_head := $node/ancestor::node//node[@end = $node/@ud:HeadPosition 
	       and @ud:Relation=("amod","app","nmod","nsubj","nsubj:pass","nummod","obj","iobj","obl","obl:agent")]
	return 
	if ($node[@ud:Relation="conj"] and exists($coord_head))
	then <headdep head="{$coord_head/@ud:HeadPosition}" dep="{$coord_head/@ud:Relation}"/>
	else ()
};

(: de onrust kan een reis vertragen of frustreren  --> obj(vertragen,reis) obj(frustreren,reis) :)
(: todo: passives ze werd ontmanteld en verkocht  su coindexed with two obj1 :)
(: done: phrases [np_i [een scoutskameraad] werd .. en _i zocht hem op] :)
(: idem: de hond was gebaseerd op Lassy en verscheen onder de naam Wirel nsubj:pass in conj1, nsubj in conj 2 :)
declare function local:distribute_dependents($node as node()) as node()* {
	let $phrase := if ($node[@rel="hd"]) then $node/.. else $node
	for $conj_head in $node/ancestor::node//node[@rel="cnj" 
	                                             and node[@rel=$phrase/@rel and not(@pt or @cat)]/@index = $phrase/@index 
	                                             and node[@rel="hd" and (@pt or @cat) and not(@ud:pos=("ADP","AUX")) and not(@cat="mwu")]]  
	                                              (: not coordination of AUX or (complex) Ps :)
    let $udRelation := local:non_local_dependency_label($phrase,($node/ancestor::node//node[@rel="cnj"]/
    	                    node[@rel=$phrase/@rel and not(@pt or @cat) and @index=$phrase/@index])[1])                                          
	where $phrase[@rel=("obj1","su","mod","pc","det") and @index]
	return <headdep head="{local:internal_head_position($conj_head)}" dep="{$udRelation}"/>
};

declare function local:enhanced_elements_to_string($headdep as node()*) as xs:string {
	let $sorted := 
		for $hd in $headdep
		order by number($hd/@head)
		return string-join(($hd/@head,$hd/@dep),":")
	return string-join($sorted,"|")

};
 
 (: Het kan voorkomen dat bepaalde gebieden en steden voor onbepaalde tijd ontoegankelijk zijn of niet verlaten kunnen worden . :)
 (: het is su of kan but controls sup of voorkomen , kunnen is aux so no big deal for control but still correct relation should be expl! :)

declare function local:enhanced_dependencies($node as node()) as node() {
	(: iobj2 control : de commissie kan de raad aanbevelen/adviseren/ X te doen :)
	(: rhd : een levend visje dat doorgeslikt moet worden :)
	let $so_index := 
	    $node/ancestor::node/node[@rel=("su","obj1","obj2") and local:internal_head_position(.) = $node/@end ]/@index
	let $rhd_index := 
	    $node/ancestor::node/node[@rel="rhd" and local:internal_head_position(.) = $node/@end ]/@index
	let $rhd_np := $node/ancestor::node[@cat="np" and node[@rel="mod"]/node[@rel="rhd"]/@index = $rhd_index]
	(: de enige _i die voldoet aan de eisen -- make sure empty heads are covered as well :) 
	let $rhdref := <headdep head="{local:internal_head_position_with_gapping($rhd_np)}" dep="ref"/>
	let $self := ( <headdep head="{$node/@ud:HeadPosition}" dep="{$node/@ud:Relation}"/> , 
		              local:anaphoric_relpronoun($node), 
		                 local:distribute_conjuncts($node),
		                     local:distribute_dependents($node) )
	
	let $enhanced := 
		if ($node[@ud:Relation=("nsubj","obj","iobj","nsubj:pass") and exists($so_index)] )
		then local:enhanced_elements_to_string(($self,
			                                      local:xcomp-control($node,$so_index),
			                                        local:upstairs-control($node,$so_index),
			                                          local:passive-vp-control($node,$so_index)) )     
		else if (exists($rhd_index)) 
		     then if (exists($rhd_np)) 
		          then local:enhanced_elements_to_string(($rhdref,
			                                           local:xcomp-control($node,$rhd_index),
			                                             local:passive-vp-control($node,$rhd_index)) )  
		          else local:enhanced_elements_to_string(($self,    (: if there is no antecedent, lets keep the basic relation :)
		          	         local:xcomp-control($node,$rhd_index),
			                   local:passive-vp-control($node,$rhd_index)) )
		     else if  ($node/@ud:HeadPosition)
		          then local:enhanced_elements_to_string($self)
		          else ""
	return 
	<node ud:Enhanced="{$enhanced}">
	        {($node/@*, 
		    for $child in $node/node return local:enhanced_dependencies($child))
            }
    </node>
} ;

(: de Amerikaanse en Pakistaanse geheime dienst 
   Alpino NF : [de amerikaanse [geh dienst]_i and Pakistaanse _i ] 
   straighforward mapping to UD leads to a conj pointing to the right, so we need to
   restore the head in its original position 
 :)

(: voorkwam dat LPF opnieuw of SGP voor het eerst in de regering zou komen  -- gapped LD :)
declare function local:fix_misplaced_heads_in_coordination($node as node()) as node() {
  let $misplaced_head := $node/ancestor::node[@cat="conj"]/node[@rel="cnj"]//
                                 node[@rel=("hd","ld") and (@pt or @cat) and @index=$node/@index]
     (: remove content from misplaced head :)
  return 
  if ($node[@rel=("hd","ld") and  @index and (@pt or @cat) and ancestor::node[@rel="cnj"] and
  	        ancestor::node[@cat="conj"]/node[@rel="cnj" and 
  	                                         descendant-or-self::node[@rel=("hd","ld") and 
  	                                                                  @index=$node/@index and 
  	                                                                  not(@cat or @pt) and 
  	                                                                  ( @begin  = ..//node[@cat or @pt]/@end 
  	                                                                  or @begin - 1 = ..//node[@cat or @pt]/@end 
  	                                                                  )
  	                                                                 ]
  	                                        ]])
  then (: create index node, dont copy content :)
       element {name($node)}  
               {($node/@index, $node/@begin, $node/@end, $node/@id, $node/@rel )} 
  else (: empty head that needs to be filled :)
       if ($node[@rel=("hd","ld","vc") and @index and not(@pt or @cat) and 
       	         ancestor::node[@rel="cnj"]  and 
  	                            ( @begin     = ..//node[@cat or @pt]/@end 
  	                              or 
  	                              @begin - 1 = ..//node[@cat or @pt]/@end 
  	                             ) 
  	                            ]
  	       and (exists($misplaced_head))) (: dont match with whd - ld cases :)
       then (: copy content :)
            element {name($node)}  
                    {$misplaced_head/@*, for $child in $misplaced_head/node return $child} (: copying id as well, but conversion never refers to this :)
       else element {name($node)}  
                    {($node/@*, for $child in $node/node 
       	                        return local:fix_misplaced_heads_in_coordination($child))
                    }
};


declare function local:add_pos_tags($node as node()) as node() {
  if ($node[@pt])
  then element {name($node)} {($node/@*, local:universal_pos_tag($node))}
  else element {name($node)} {($node/@*, for $child in $node/node return local:add_pos_tags($child))}
};

declare function local:universal_pos_tag($node as element(node)) as attribute() {
  let $PT := $node/@pt
  let $REL := $node/@rel
  return 
  attribute ud:pos {
         if ($PT eq 'let') then 
           (: SYM is a brute force solution to avoid interaction with udapy FixPunct
              relevant for let in mwe cases and let as part of names :) 
           if ($REL eq '--' ) then 
            	if ( $node/../node[not(@pt='let')] ) then 'PUNCT'
            	else if ( $node[../node/@begin < @begin] ) 
                 	then 'PUNCT'
                 	else 'SYM'  
              else 'SYM'

    else if ($PT eq 'adj') then 'ADJ'
    else if ($PT eq 'bw')  then 'ADV'
    else if ($PT eq 'lid') then 'DET'
    else if ($PT eq 'n')   then
      if ($node/@ntype eq 'eigen') then  'PROPN' else 'NOUN'

    else if ($PT eq 'spec') then
      if ( $node/@spectype eq 'deeleigen') then 'PROPN'
      else if ($node/@spectype eq 'symb')  then  'SYM'  
      else 'X'  (: afk vreemd afgebr enof meta :)
    else if ($PT eq 'tsw') then 'INTJ'
    else if ($PT eq 'tw')  then 
         if ($node/@numtype="rang") then 'ADJ'
         else 'NUM'
    else if ($PT eq 'vz')  then 'ADP'  (: v2: do not use PART for SVPs and complementizers :)
    else if ($PT eq 'vnw') 
         then if ($REL eq 'det' and not($node/@vwtype="bez") )  then 'DET'
         else if ($node/@pdtype eq 'adv-pron' ) then 'ADV' 
         else 'PRON'
    else if ($PT eq 'vg') then
      if ($node/@conjtype eq 'neven')  then  'CCONJ'  else 'SCONJ' (: V2: CONJ ==> CCONJ :)
    else if ($PT eq 'ww') then 
      if (local:auxiliary($node) eq 'verb') then  'VERB' else  'AUX' (: v2: cop and aux:pass --> AUX  (already in place in v1?) :)
    else  'ERROR_NO_POS_FOUND'
    }
};

declare function local:auxiliary($nodes as element(node)*) as xs:string
{ if (count($nodes) = 0)
  then "ERROR_AUXILIARY_FUNCTION_TAKES_EXACTLY_ONE_ARG"
  else local:auxiliary1($nodes[1])
} ;

declare function local:auxiliary1($node as element(node)) as xs:string
{ if ($node[@pt="ww" and @rel="hd" and not(../node[@rel=("obj1","se","vc")]) and
	        (: ud documentation suggests 1 cop per lg, van Eynde suggests much more, compromise: the traditional ones :)
	        @lemma=("zijn","lijken","blijken","blijven","schijnen","heten","voorkomen","worden","dunken") and 
                 ( contains(@sc,'copula') or
                   contains(@sc,'pred')   or   
                   contains(@sc,'cleft')  or 
                   ../node[@rel="predc"] 
                 ) ] )                                
  then "cop"    
  else if ( $node[@pt="ww" and @rel="hd" and @lemma=("zijn","worden") and 
                  ( @sc="passive"  or 
                  	 ( ../node[@rel="vc"] and 
                        ( ../node[@rel="su"]/@index = ../node[@rel="vc"]/node[@rel="obj1"]/@index or 
                          ../node[@rel="su"]/@index = ../node[@rel="vc"]/node[@rel="cnj"]/node[@rel="obj1"]/@index or
                          ../node[@rel="vc" and not(@pt or @cat)]/@index = 
                              ancestor::node//node[@rel="vc" and node[@rel="obj1"]/@index = $node/../node[@rel="su"]/@index]/@index
                         or  not(../node[@rel="su"]) 
                         )  
                     ) 
                  ) ] )           
  then "aux:pass"    
  (: krijgen passive with iobj control :)
  else if ( $node[@pt="ww" and @rel="hd" and @lemma="krijgen" and
  	              ( ../node[@rel="su"]/@index = ../node[@rel="vc"]/node[@rel="obj2"]/@index or 
                    ../node[@rel="su"]/@index = ../node[@rel="vc"]/node[@rel="cnj"]/node[@rel="obj2"]/@index
                  )])
  then "aux:pass" 
  (: alpino has no principled distinction between AUX and VERB, should be TAME verbs semantically, we follow ENGLISH :)
  else if ($node[@pt="ww" and @rel="hd" and not(../node[@rel="predc"]) and  (: hij heeft als opdracht stammen uit elkaar te houden  :)
                 ( starts-with(@sc,'aux') or
                   ( ../node[@rel="vc"  and   
                              ( @cat=("ppart","inf","ti") or 
                                ( @cat="conj" and node[@rel="cnj" and @cat=("ppart","ti","inf")] ) or
                                ( @index and not(@pt or @cat))  (: dangling aux in gapped coordination :)
                              )  
                            ]   and  
                     @lemma=("blijken","hebben","hoeven","kunnen","moeten","mogen","zijn","zullen") 
                   ) 
                 ) 
               ])                           
  then "aux"    
  else if ($node[@pt="ww"] )
  then "verb"
  else "ERROR_NO_VERB"
};

declare function local:add_features($node as node()) as node() {
  if      ($node[@ud:pos="NOUN" or @ud:pos="PROPN"])
  then    element {name($node)} {($node/@*, local:nominal_features($node))}
  
  else if ($node[@ud:pos="ADJ"]) 
  then    element {name($node)} {($node/@*, local:adjective_features($node))}
  
  else if ($node[@ud:pos="PRON"] )
  then    element {name($node)} {($node/@*, local:pronoun_features($node))}
       
  else if ($node[@ud:pos="VERB" or @ud:pos="AUX"] )
  then    element {name($node)} {($node/@*, local:verbal_features($node))}
       
  else if ($node[@ud:pos="DET"] )
  then    element {name($node)} {($node/@*, local:determiner_features($node))}  
  

  else if ($node[@ud:pos="X"] )
  then    element {name($node)} {($node/@*, local:special_features($node))}  
  
       
  else    element {name($node)} {($node/@*, for $child in $node/node return local:add_features($child))}
};

declare function local:special_features($node as element(node)) as attribute()+ { 
	( attribute ud:Foreign {
		if ($node/@spectype="vreemd")
		then "Yes"
		else "null"
	}
	,
	  attribute ud:Abbr {
	  	if ($node/@spectype="afk")
	  	then "Yes"
	  	else "null"
	  }
    )
};

declare function local:nominal_features($node as element(node)) as attribute()+ {
  let $GENUS := $node/@genus
  let $GETAL := $node/@getal
  let $DEGREE := $node/@graad
  return 
  ( attribute ud:Gender {
    if      ($GENUS eq 'zijd')  then 'Com' 
    else if ($GENUS eq 'onz')   then 'Neut'
    else if ($GENUS eq 'genus') then 'Com,Neut'
    else if ($GENUS)            then 'ERROR_IRREGULAR_GENDER'
    else                             'null'
    }
  ,
  attribute ud:Number {
    if      ($GETAL eq 'ev') then 'Sing' 
    else if ($GETAL eq 'mv') then 'Plur'
    else if ($GETAL)         then 'ERROR_IRREGULAR_NUMBER'
    else                          'null'
    }
  ,
  attribute ud:Degree {
    if      ($DEGREE eq 'dim' )  then 'null'
    else if ($DEGREE eq 'basis') then 'null'
    else if ($DEGREE)            then 'ERROR_IRREGULAR_DEGREE'
    else                              'null'
    }
  )
};

declare function local:adjective_features($node as element(node)) as attribute()+ {
  let $GRAAD := $node/@graad
  return 
  attribute ud:Degree
    {
    if      ($GRAAD eq 'basis') then 'Pos' 
    else if ($GRAAD eq 'comp')  then 'Cmp'
    else if ($GRAAD eq 'sup')   then 'Sup'
    else if ($GRAAD eq 'dim')   then 'Pos' (: netjes :) 
    else if ($GRAAD)            then 'ERROR_IRREGULAR_DEGREE'
    else                             'null'
    }
};

declare function local:pronoun_features($node as element(node)) as attribute()+ {
  let $VWTYPE  := $node/@vwtype
  let $PERSOON := $node/@persoon
  let $NAAMVAL := $node/@naamval
  return 
  ( attribute ud:PronType {
    if      ($VWTYPE eq 'pers')  then 'Prs' 
    else if ($VWTYPE eq 'refl')  then 'Prs'
    else if ($VWTYPE eq	'pr')    then 'Prs'
    else if ($VWTYPE eq	'bez')   then 'Prs'
    else if ($VWTYPE eq 'recip') then 'Rcp'
    else if ($VWTYPE eq 'vb')    then 'Int'
    else if ($VWTYPE eq 'aanw')  then 'Dem'
    else if ($VWTYPE eq 'onbep') then 'Ind'
    else if ($VWTYPE eq 'betr')  then 'Rel'
    else if ($VWTYPE eq 'excl')  then 'null' (: occurs only once :)
    else if ($VWTYPE)            then 'ERROR_IRREGULAR_PRONTYPE'
    else                              'null'
    }
    ,
    attribute ud:Reflex {
    if ($VWTYPE eq 'refl') 	then 'Yes'
    else 			     'null'
    }
    ,
    attribute ud:Poss {
    if ($VWTYPE eq 'bez') 	then 'Yes'
    else 			     'null'
    }
    ,
    attribute ud:Person {
    if      ($PERSOON eq '1')       then '1'
    else if ($PERSOON eq '2')       then '2'
    else if ($PERSOON eq '2b')      then '2'
    else if ($PERSOON eq '2v')      then '2'
    else if ($PERSOON eq '3')       then '3'
    else if ($PERSOON eq '3o')      then '3'
    else if ($PERSOON eq '3v')      then '3'
    else if ($PERSOON eq '3p')      then '3'
    else if ($PERSOON eq '3m')      then '3'
    else if ($PERSOON eq 'persoon') then 'null'
    else if ($PERSOON)        then 'IERROR_RREGULAR_PERSON'
    else                           'null'
    }
    ,
    attribute ud:Case {
    if    ($NAAMVAL eq 'nomin')  then 'Nom'
    else if ($NAAMVAL eq 'obl')  then 'Acc'
    else if ($NAAMVAL eq 'gen')  then 'Gen'
    else if ($NAAMVAL eq 'dat')  then 'Dat' (: van dien aard :) 
    else if ($NAAMVAL eq 'stan') then 'null'
    else if ($NAAMVAL)           then 'ERROR_IRREGULAR_CASE'
    else                              'null'
    }
  )
};

declare function local:verbal_features($node as element(node)) as attribute()+ {
  let $WVORM  := $node/@wvorm
  let $PVTIJD := $node/@pvtijd
  let $PVAGR  := $node/@pvagr
  return 
  ( attribute ud:VerbForm   {
    if      ($WVORM eq 'pv')  then 'Fin' 
    else if ($WVORM eq 'inf') then 'Inf'
    else if ($WVORM eq 'vd')  then 'Part'
    else if ($WVORM eq 'od')  then 'Part'
    else if ($WVORM)          then 'ERROR_IRREGULAR_VERBFORM'
    else                           'null'
    }
    ,
    attribute ud:Tense {
    if      ($PVTIJD eq 'verl') then 'Past'
    else if ($PVTIJD eq 'tgw')  then 'Pres'
    else if ($PVTIJD eq 'conj') then 'Pres'
    else if ($PVTIJD)           then 'ERROR_IRREGULAR_TENSE'
    else                             'null'
    }
    ,
    attribute ud:Number {
    if      ($PVAGR eq 'ev' )   then 'Sing'
    else if ($PVAGR eq 'met-t') then 'Sing'
    else if ($PVAGR eq 'mv')    then 'Plur'
    else if ($PVAGR)            then 'ERROR_IRREGULAR_NUMBER'
    else                             'null'
    }
  )
};

declare function local:determiner_features($node as element(node)) as attribute()+ {
  let $LWTYPE := $node/@lwtype
  return 
  attribute ud:Definite
    {
    if      ($LWTYPE eq 'bep')   then 'Def' 
    else if ($LWTYPE eq 'onbep') then 'Ind'
    else if ($LWTYPE)            then 'ERROR_IRREGULAR_DEFINITE'
    else                              'null'
    }
};


declare function local:add_dependency_relations($node as node()) as node() {
  if ($node[@pt])
  then element {name($node)} {($node/@*, local:dependency_relation($node))}
  else element {name($node)} {($node/@*, for $child in $node/node return local:add_dependency_relations($child))}
};

declare function local:dependency_relation($node as element(node)) as attribute()+ {
  ( attribute ud:Relation     { local:dependency_label($node) }  ,
    attribute ud:HeadPosition { local:external_head_position($node) } 
  )
} ;

(: used for debugging, but also nice as readible alternative for selecting by [1] or by means of @begin position :)
declare function local:leftmost($nodes as element(node)*) as element(node) {
	let $sorted :=  for $node in $nodes
	                order by number($node/@begin)
	                return $node
	return
	    $sorted[1]
};

declare function local:following-cnj-sister($node as element(node)) as element(node)
{ let $annotated-sisters :=
      for $sister in $node/../node[@rel="cnj"]
      return
         <begin-node begin="{local:begin-position-of-first-word($sister)}">
           {$sister}
         </begin-node>
       
  let $sorted-sisters :=
      for $sister in $annotated-sisters
      (: where $sister[number(@begin) > $node/number(@begin)] :)
      order by $sister/number(@begin)
      return $sister
  return 
      if  ($sorted-sisters[number(@begin) > $node/number(@begin)] ) 
      then ($sorted-sisters[number(@begin) > $node/number(@begin)]/node)[1]
      else $sorted-sisters[1]/node

};

(: recompute begin positions based on actual words in string only :) 
declare function local:begin-position-of-first-word($node as element(node)) as xs:string 
{ let $words := 
         for $leaf in $node//node[@word]
         order by $leaf/number(@begin)
         return
            $leaf
  return
      if ($node[@word]) then $node/@begin 
      else if ($words) then $words[1]/@begin
           else "100"
};

declare function local:internal_head_position($nodes as element(node)*) as xs:string
{ if ( count($nodes) = 1 )
  then local:internal_head_position1($nodes[1])
  else if ( count($nodes) = 0 )
       then "ERROR_NO_INTERNAL_HEAD_POSITION_FOUND"
       else "ERROR_MORE_THAN_ONE_INTERNAL_HEAD_POSITION_FOUND"
} ;

declare function local:internal_head_position1($node as element(node)) as xs:string
{ if      ($node[@cat="pp"])
  then    if ($node/node[@rel="hd" and @pt=("bw","n")] )  (: n --> TEMPORARY HACK to fix error where NP is erroneously tagged as PP :)
          then $node/node[@rel="hd"]/@end
          else if ($node/node[@rel=("obj1","pobj1","se")]) 
               then local:internal_head_position($node/node[@rel=("obj1","pobj1","se")][1])
               else if ($node/node[@rel="hd" and @cat="mwu"])  (: mede [op grond hiervan] :)
                    then local:internal_head_position($node/node[@rel="hd"] )
                    else local:internal_head_position( $node/node[1] )    
  
  else if ($node[@cat="mwu"] ) 
  then    $node/node[@rel="mwp" and not(../node/number(@begin) < number(@begin))]/@end
  
  else if ($node[@cat="conj"])
  then    local:internal_head_position(local:leftmost($node/node[@rel="cnj"]))
  
  else if ( $node/node[@rel="predc"] ) 
       then if ($node/node[@rel="hd" and @ud:pos="AUX"])
            then local:internal_head_position($node/node[@rel="predc"])
            else if ( not($node/node[@rel="hd"]) )      (: cases where copula is missing by accident (ungrammatical, not gapping) :)
                 then local:internal_head_position($node/node[@rel="predc"])
                 else local:internal_head_position($node/node[@rel="hd"])
  
  else if ($node[node[@rel="vc"] and 
                 node[@rel="hd" and 
                      ( @ud:pos="AUX" or 
                        $node/ancestor::node[@rel="top"]//node[@ud:pos="AUX"]/@index = @index
                       )
                     ] 
                ]
          )
  then    local:internal_head_position($node/node[@rel="vc"])
   
  else if ( $node/node[@rel="hd"]) 
  then local:internal_head_position($node/node[@rel="hd"][1])

  else if ( $node/node[@rel="body"]) 
  then    local:internal_head_position($node/node[@rel="body"][1])
  
  else if ( $node/node[@rel="dp"]) 
  then    local:internal_head_position(local:leftmost($node/node[@rel="dp"]))
       (: sometimes co-indexing leads to du's starting at same position ... :)

  else if ( $node/node[@rel="nucl"]) 
  then    local:internal_head_position($node/node[@rel="nucl"][1])

  else if ( $node/node[@cat="du"]) (: is this neccesary at all? , only one referring to cat, and causes problems if applied before @rel=dp case... :)
  then    local:internal_head_position($node/node[@cat ="du"][1])

  else if ( $node[@word] )
  then    $node/@end
  
  (: distinguish empty nodes due to gapping/RNR from nonlocal dependencies 
     use 'empty head' as string to signal precence of gapping/RNR :)
  else if ($node[@index and not(@word or @cat)] )
     then if ($node/ancestor::node[@rel="top"]//node[@rel=("whd","rhd") and @index = $node/@index and (@word or @cat)] )
          then local:internal_head_position($node/ancestor::node[@rel="top"]//node[@index = $node/@index and (@word or @cat)] )
          else "empty head" 
    
  else    'ERROR_NO_INTERNAL_HEAD'
};

declare function local:internal_head_position_with_gapping($node as element(node)*) as xs:string {
	let $hd_pos := local:internal_head_position($node)
	return  if  ($hd_pos eq "empty head") 
	        then local:internal_head_position_of_gapped_constituent($node)
	        else $hd_pos
};

declare function local:internal_head_position_of_gapped_constituent($node as element(node)) as xs:string {
  if ($node/node[@rel="hd" and (@pt or @cat)])
   then local:internal_head_position_with_gapping($node/node[@rel="hd"])  (: auxiliaries, prepositions, ... :)
  else if ( $node/node[@rel="su" and (@pt or @cat)] ) 
	then local:internal_head_position_with_gapping($node/node[@rel="su"]) (: 44 van 87 in lassysmall:)
  else if ($node[@rel="vc" and ../node[@rel="su" and (@pt or @cat)]] ) 
    (: subject realized inside the vc = funny serialization :)
    then local:internal_head_position_with_gapping($node/../node[@rel="su"])
  else if ( $node/node[@rel="obj1" and (@pt or @cat)] )
	then local:internal_head_position_with_gapping($node/node[@rel="obj1"])  
  else if ( $node/node[@rel="predc" and (@pt or @cat)] )
	then local:internal_head_position_with_gapping($node/node[@rel="predc"])
  else if ( $node/node[@rel="vc" and (@pt or @cat)] )
    then local:internal_head_position_with_gapping($node/node[@rel="vc"][1])  
  else if ( $node/node[@rel="pc" and (@pt or @cat)] )
    then local:internal_head_position_with_gapping($node/node[@rel="pc"][1]) 
  else if ( $node/node[@rel="mod" and (@pt or @cat)] )
	then local:internal_head_position_with_gapping(($node/node[@rel="mod" and (@pt or @cat)])[1]) 
  else if ( $node/node[@rel="app" and (@pt or @cat)] )
    then local:internal_head_position_with_gapping(($node/node[@rel="app" and (@pt or @cat)])[1])
  else if ( $node/node[@rel="det" and (@pt or @cat)] )
    then local:internal_head_position_with_gapping(($node/node[@rel="det" and (@pt or @cat)])[1])
  else if ( $node/node[@rel="body" and (@pt or @cat)] )
    then local:internal_head_position_with_gapping(($node/node[@rel="body" and (@pt or @cat)])[1]) 
  else if ( $node/node[@rel="cnj" and (@pt or @cat)] )
    then local:internal_head_position_with_gapping(($node/node[@rel="cnj" and (@pt or @cat)])[1])
  else  "ERROR_NO_INTERNAL_HEAD_IN_GAPPED_CONSTITUENT"
};

declare function local:external_head_position($nodes as element(node)*) as xs:string
{ if (count($nodes) = 0 ) 
  then "ERROR_EXTERNAL_HEAD_MUST_HAVE_ONE_ARG"
  else local:external_head_position1($nodes[1]) 
} ;

declare function local:external_head_position1($node as element(node)) as xs:string
{  if ($node[@rel="hd" and (@ud:pos="ADP" or ../@cat="pp") ] )  (: vol vertrouwen :)
    then if ($node/../node[@rel="predc"] ) (: met als titel :)
          then local:internal_head_position(($node/../node[@rel="predc"])[1])
          else if ($node/../node[@rel= ("obj1","vc","se","me") and (@pt or @cat)] ) 
               (: adding pt/cat enough for gapping cases? :) 
                then local:internal_head_position_with_gapping(($node/../node[@rel= ("obj1","vc","se","me")])[1])
                else if ($node/../node[@rel="pobj1"] )
                      then local:internal_head_position(($node/../node[@rel="pobj1"])[1] )
                        (: in de eerste rond --> typo in LassySmall/Wiki , binnen en [advp later buiten ]:)
                      else local:external_head_position($node/..)

   else if ($node[@rel="hd" and starts-with(local:auxiliary($node),'aux')] ) (: aux aux:pass  :)
          then if ($node/../node[@rel=("vc","predc") and (@pos or (@cat and node[@pt or @cat]))])  (: skip vc with just empty nodes :)
    	        then local:internal_head_position_with_gapping(($node/../node[@rel="vc"])[1])
    			else local:external_head_position($node/..)  (: gapping, but does it ever occur with aux?? :)

   else if ($node[@rel="hd" and local:auxiliary($node) eq 'cop'] ) 
          then if ($node/../node[@rel="predc" and (@pos or @cat)])
    	        then local:internal_head_position_with_gapping(($node/../node[@rel="predc"])[1])
    			else if ($node/../node[@rel="predc"]/@index = $node/ancestor::node/node[@rel=("rhd","whd")]/@index)
    			      then local:internal_head_position($node/ancestor::node/node[@rel=("rhd","whd") and @index = $node/../node[@rel="predc"]/@index] )
    			      else local:external_head_position($node/..)  (: gapping, but could it??:)

   else if ($node[@rel=("hd","nucl","body") ] ) 
    then if ($node/../node[@rel="hd"]/number(@begin) < $node/number(@begin) ) 
          then local:internal_head_position($node/../node[@rel="hd" and number(@begin) < $node/number(@begin)] ) (: dan moet je moet :)
          else local:external_head_position($node/..)

   else if ( $node[@rel="predc"] ) 
    then if   ($node[../node[@rel=("obj1","se","vc")] and ../node[@rel="hd" and (@pt or @cat)]] )
          then if ( $node/../node[@rel="hd" and @ud:pos="ADP"] ) 
                then local:external_head_position($node/..) (: met als presentator Bruno W , met als gevolg [vc dat ...]:)
                else local:internal_head_position($node/../node[@rel="hd"])
          else if  ( $node/..[@cat=("np","ap") and node[@rel="hd" and (@pt or @cat) and not(@ud:pos="AUX") ]  ]  )  
                          (: reduced relatives , make sure head is not empty (ellipsis) :)
                          (: also : ap with predc: actief als schrijver :)
                then local:internal_head_position($node/../node[@rel="hd"])
                else if ($node/../node[@rel="hd" and (@pt or @cat) and not(@ud:pos=("AUX","ADP"))] )  (: [met als titel] -- obj1/vc missing :)
                	 then local:internal_head_position($node/../node[@rel="hd"])
                	 else local:external_head_position($node/..) (: covers gapping as well? :)

   else if ( $node[@rel=("obj1","se","me") and (../@cat="pp" or ../node[@ud:pos="ADP" and @rel="hd"])] )
    then if ($node/../node[@rel="predc"] ) 
          then local:internal_head_position($node/../node[@rel="predc"])
          else local:external_head_position($node/..)
  
   else if ( $node[@rel="pobj1" and (../@cat="pp" or ../node[@ud:pos="ADP" and @rel="hd"])] )
    then if ($node/../node[@rel="vc"])  
          then local:internal_head_position($node/../node[@rel="vc"])
          else local:external_head_position($node/..)
   
   else if ($node[@rel="mod" and not(../node[@rel=("obj1","pobj1","se","me")]) and (../@cat="pp" or ../node[@rel="hd" and @ud:pos="ADP"])])   (: mede op grond hiervan :)
    (: daarom dus :)
         then if ($node/../node[@rel=("hd","su","obj1","vc") and (@pt or @cat)] )
               then local:internal_head_position_with_gapping($node/..)  
               else local:external_head_position($node/..) (: gapping :)
  
 
  else if ($node[@rel=("cnj","dp","mwp")])
   then if ( deep-equal($node,local:leftmost($node/../node[@rel=("cnj","dp","mwp")])) )  
         then local:external_head_position($node/..)
         else if ($node[@rel="cnj"]) 
              then local:head_position_of_conjunction($node)
              else local:internal_head_position_with_gapping($node/..)
  
  else if ($node[@rel="cmp" and ../node[@rel="body"]])
   then local:internal_head_position_with_gapping($node/../node[@rel="body"][1])      
  
  else if ($node[@rel="--" and @cat] )
  	then if ($node[@cat="mwu"]) 
          then if ($node/../node[@cat and not(@cat="mwu")]  )    (: fix for multiword punctuation in Alpino output :)
                then local:internal_head_position($node/../node[@cat and not(@cat="mwu")][1])
                else local:external_head_position($node/..)
    else local:external_head_position($node/..)

  else if ( $node[@rel="--" and @ud:pos] )
   then if ($node[@ud:pos = ("PUNCT","SYM","X","CONJ","NOUN","PROPN","NUM","ADP","ADV","DET","PRON") 
                  and ../node[@rel="--" and 
                              not(@ud:pos=("PUNCT","SYM","X","CONJ","NOUN","PROPN","NUM","ADP","ADV","DET","PRON")) ] 
                 ] ) 
         then local:internal_head_position_with_gapping($node/../node[@rel="--" and not(@ud:pos=("PUNCT","SYM","X","CONJ","NOUN","ADP","ADV","DET","PROPN","NUM","PRON"))][1])
         else if ( $node/../node[@cat]  ) 
               then local:internal_head_position($node/../node[@cat][1])
               else if ($node[@ud:pos="PUNCT" and count(../node) > 1]) 
                     then if ($node/../node[not(@ud:pos="PUNCT")] )
                           then local:internal_head_position($node/../node[not(@ud:pos="PUNCT")][1])
                           else if ( deep-equal($node,local:leftmost($node/../node[@rel="--" and (@cat or @pt)]) ) )
                                 then local:external_head_position($node/..)
                                 else "1" (: ie end of first punct token :)
                     else if ($node/..) then local:external_head_position($node/..)
    else "ERROR_NO_HEAD_FOUND"
  
  else if ($node[@rel=("dlink","sat","tag")])
   then if ($node/../node[@rel="nucl"])
         then local:internal_head_position($node/../node[@rel="nucl"])
         else "ERROR_NO_EXTERNAL_HEAD"
   
  else if ($node[@rel="vc"]) 
    then if ($node/../node[@rel="hd" and 
                           ( @ud:pos="AUX" or 
                             $node/ancestor::node[@rel="top"]//node[@ud:pos="AUX"]/@index = @index
                           )
                       ]
                  and not($node/../node[@rel="predc"]) )
          then local:external_head_position($node/..)
          else if ($node/../@cat="pp") (: eraan dat .. :)
                 then local:external_head_position($node/..)
                 else if ($node/../node[@rel=("hd","su") and (@pt or @cat)] )
                      then local:internal_head_position_with_gapping($node/..)
                      else local:external_head_position($node/..)
            
  else if ($node[@rel="whd" or @rel="rhd"]) 
   then if ($node[@index])
         then local:external_head_position( ($node/../node[@rel="body"]//node[@index = $node/@index ])[1] )
         else local:internal_head_position($node/../node[@rel="body"])
    
(: we need to select the original node and not the result of 
   following-cnj-sister, as that has no global context 
   and global context is needed where the hd is an index node...
   unfortunately, nodes are completely identical in some 
   elliptical cases, select last() as brute force solution :)
  else if ($node[@rel="crd"])
   then local:internal_head_position_with_gapping(
   	           $node/../node[@rel="cnj" and 
         	                 @begin=local:following-cnj-sister($node)/@begin and 
         	                 @end=local:following-cnj-sister($node)/@end
         	                ][last()] )
       
  else if ($node[@rel="su"]) 
   then if ($node/../node[@rel="hd" and (@pt or @cat)]) (: gapping :)
        then local:internal_head_position_with_gapping($node/..) (: ud head could still be a predc :)
         (: only for 1 case where verb is missing -- die eigendom ... (no verb)) :)
        else if ($node[../node[@rel="predc"] and not(../node[@rel="hd"])] )
             then local:internal_head_position($node/../node[@rel="predc"]) 
         	 else local:external_head_position($node/..) (: this probably does no change anything, as we are still attaching to head of left conjunct :)
  
  else if ($node[@rel="obj1"]) 
   then if ($node/../node[@rel=("hd","su") and (@pt or @cat)]) (: gapping, as su but now su could be head as well :)
         then local:internal_head_position_with_gapping($node/..) 
         else local:external_head_position($node/..) 

  else if ($node[@rel="pc"]) 
   then if ($node/../node[@rel=("hd","su","obj1") and (@pt or @cat)]) (: gapping, as su but now su could be head as well :)
         then local:internal_head_position_with_gapping($node/..) 
         else local:external_head_position($node/..) 

  else if ($node[@rel="mod"]) 
   then if ($node/../node[@rel=("hd","su","obj1","pc","predc","body") and (@pt or @cat)]) (: gapping, as su but now su or obj1  could be head as well :)
         then local:internal_head_position_with_gapping($node/..) 
         else if ($node/../node[@rel="mod" and (@cat or @pt)])
               then if  (deep-equal($node,local:leftmost($node/../node[@rel="mod" and (@pt or @cat)])) ) (: gapping with multiple mods :)
                     then local:external_head_position($node/..) 
                     else local:internal_head_position_with_gapping($node/..) 
               else if ( $node/../../node[@rel="su" and (@pt or @cat)]  )  (: an mod in an otherwise empty tree (after fixing heads in conj) :)
                    then local:internal_head_position($node/../../node[@rel="su"])
                    else local:external_head_position($node/..) (: an empty mod in an otherwise empty tree 
                                                              -- mod is co-indexed with rhd, rest is elided, 
                                                              LassySmall4/wiki-7064/wiki-7064.p.28.s.3.xml :)

else if ($node[@rel=("app","det")]) 
   then if ($node/../node[@rel=("hd","mod") and (@pt or @cat)]) (: gapping with an app (or a det)! :)
         then local:internal_head_position_with_gapping($node/..) 
         else local:external_head_position($node/..) 
             
  else if ($node[@rel="top"])   
   then "0"
  
  else if ( $node[not(@rel="hd")] )
    then    local:internal_head_position_with_gapping($node/..)
  
  else    'ERROR_NO_EXTERNAL_HEAD'
} ;


 (: brute force method to be compliant with conj points to the left rule: :)
  (: if interhdpos($node) < internalhdpos($node/..) then do something ad hoc :)
  (: because even fixing misplaced heads fails in cases like 
De vertegenwoordigers van het gas- en elektriciteitsbedrijf zouden vandaag en die van de mijnwerkers overmorgen hun stakingsplannen bekend maken , terwijl die van de PTT een voorbericht hebben gegeven voor een staking op 2 oktober .
 Het front der activisten vertoont dan wel een beeld van lusteloosheid , " aan de basis " is en wordt toch veel werk verzet . 
:)
declare function local:head_position_of_conjunction($node as element(node)) as xs:string 
{ let $internal_head := local:internal_head_position_with_gapping($node)
  let $leftmost_conj_daughter := local:leftmost($node/../node[@rel="cnj"])
  let $leftmost_internal_head := local:internal_head_position_with_gapping($leftmost_conj_daughter)
  let $endpos_of_leftmost_conj_constituents := 
  		for $e in $leftmost_conj_daughter/node/@end
  		where number($e) < number($internal_head) 
  		order by number($e)
  		return $e
  return 
  if (number($leftmost_internal_head) < number($internal_head))  (: business as usual :)
  then $leftmost_internal_head    
  else if ( $endpos_of_leftmost_conj_constituents )
       then $endpos_of_leftmost_conj_constituents[last()]
       else ( $leftmost_conj_daughter/node/@end)[1]  (: this should not happen really -- give error msg? :)
 
};

declare function local:dependency_label($node as element(node)) as xs:string
{   if      ($node/..[@cat="top" and @end="1"])     then "root" 
    else if ($node[@rel="app"])
         then if ($node/../node[@rel="hd" and (@pt or @cat)]) 
               then "appos"
               else if ($node/../node[@rel="mod" and (@pt or @cat)])
                    then "orphan"
                    else local:dependency_label($node/..)
    else if ($node[@rel="cmp"])                     then "mark"
    else if ($node[@rel="crd"])                     then "cc"
    else if ($node[@rel="me" and not(../node[@ud:pos="ADP"]) ])   then local:determine_nominal_mod_label($node)
    else if ($node[@rel="obcomp"])                  then "advcl"
    else if ($node[@rel="obj2"]) 
     then if ($node[@cat="pp"]) then "obl" else "iobj"
    else if ($node[@rel="pobj1"])                   then "expl"
    else if ($node[@rel="predc"]) 
     then if ( $node/../node[@rel=("obj1","se") and (@pt or @cat)] or $node/../node[@rel="hd" and (@pt or @cat) and not(@ud:pos="AUX")] ) 
           then "xcomp"
           else local:dependency_label($node/..) (: covers gapping cases where predc is promoted to head as well :)
         (: hack for now: de keuze is gauw gemaakt :)
         (: was amod, is this more accurate?? :)
         (: examples of secondary predicates under xcomp suggests so :)
    else if ($node[@rel="se"])                      then "expl:pv"
    else if ($node[@rel="su"])  					
      then if ($node[../@rel="cnj" and ../node[@rel="hd" and not(@pt or @cat)]] ) (: gapping :)
           then local:dependency_label($node/..)
           else if ($node[../@rel="vc" and ../node[@rel="hd" and not(@pt or @cat)] 
           	                and ../..[@rel="cnj" and node[@rel="hd" and not(@pt or @cat)]]] ) (: gapping with subj downstairs :)
           (: In 1909 werd de persoonlijke dienstplicht ingevoerd en in 1913 de algemene persoonlijke dienstplicht .
              [ hd_i su_j vc [ hd_k [_j pers dienstplicht ] :)
                then local:dependency_label($node/../..)
                else local:subject_label($node)
    else if ($node[@rel="sup"])                     then "expl"
    else if ($node[@rel="svp"])                     then "compound:prt"  (: v2: added prt extension:)
    else if (local:auxiliary($node) eq 'aux:pass')  then 
             if ($node[../node[@rel="su" and not(@pt or @cat)] and 
             	../node[@rel="vc" and not(@pt or @cat)] and
             	../@rel="cnj"] ) 
             then "conj" 
             else "aux:pass"                                                
    else if (local:auxiliary($node) eq 'aux')       then "aux"    
    else if (local:auxiliary($node) eq 'cop')       then "cop"    

    else if ( $node[@rel="det"] ) 
         then if ($node/../node[@rel="hd" and (@pos or @cat)])
              then local:det_label($node)
              else if ($node/../node[@rel="mod" and (@pt or @cat)] ) (: gapping :)
                   then "orphan"
                   else local:dependency_label($node/..) (: gapping :)
              
    else if ($node[@rel=("obj1","me")] )
          then if ( $node/../@cat="pp" or $node/../node[@rel="hd" and @ud:pos="ADP"]) (: vol vertrouwen , heel de geschiedenis door (cat=ap!) :)
                then local:dependency_label($node/..)  
	            else if ($node[@index = ../../node[@rel="su"]/@index ] )
                    then "nsubj:pass"  (: trees where su (with extraposed material) is spelled out at position of obj1  :)
                    else if ($node/../node[@rel="hd" and (@pt or @cat)] )
                           then "obj"
                           else if ($node/../node[@rel="su" and (@pt or @cat)] )
                                then "orphan"
                                else local:dependency_label($node/..) (: gapping :)
						    
    else if ($node[@rel="mwp"])
          then if ($node[@begin = ../@begin])
                then local:dependency_label($node/..)
			    else if ( $node/../node[@ud:pos="PROPN"]) 
			          then "flat:name"  (: v2: name --> flat:name :)
			          else "fixed"   (: v2 mwe-> fixed :)
						    
    else if ($node[@rel="cnj"])    
          then if   (deep-equal($node,$node/../node[@rel="cnj"][1]))
                then local:dependency_label($node/..)
                else "conj"       
              
     else if ($node[@rel="dp"])    
           then if   (deep-equal($node,local:leftmost($node/../node[@rel="dp"])) )
                 then local:dependency_label($node/..)
                 else "parataxis"      

    else if ($node[@rel=("tag","sat")] )           then "parataxis"
    else if ($node[@rel="dlink"])                  then "mark"
    else if ($node[@rel="nucl"])                   then  local:dependency_label($node/..)
    					    					    						   					    
    else if ($node[@rel="vc"] ) 
         then if ($node/../node[@rel="hd" and @ud:pos=("AUX","ADP")] and not($node/../node[@rel="predc"]) )
              then local:dependency_label($node/..)
              else if ($node/../node[@rel="hd" and (@pt or @cat)])
                   then if ($node/node[@rel="su" and @index and not(@word or @cat)] or
                   	        $node[@cat="ti"]/node[@rel="body"]/node[@rel="su" and @index and not(@word or @cat)] or
                   	        $node[@cat="oti"]/node[@cat="ti"]/node[@rel="body"]/node[@rel="su" and @index and not(@word or @cat)]
                   	        )
                        then "xcomp"                                    
                       	else if ($node/../@cat="np") 
                       	     then "acl"               (: v2: clausal dependents of nouns always acl :)
                       	     else "ccomp"    
                    else if ($node/../node[@rel=("su","obj1") and (@pt or @cat)] )
                           then "orphan"
                           else local:dependency_label($node/..) (: gapping :)
    
    else if ($node[@rel=("mod","pc","ld") and ../@cat="np"])  (: [detp niet veel] meer :) 
         (: modification of nomimal heads :)
         (: pc and ld occur in nominalizations :)
         then if ($node/../node[@rel="hd" and (@pt or @cat)])  
              then local:mod_label_inside_np($node)
              else if (deep-equal($node,local:leftmost($node/../node[@rel="mod" and (@pt or @cat)]))) (: gapping with multiple mods :)
                   then local:dependency_label($node/..) (: gapping, where this mod is the head :)
                   else "orphan"

    else if ($node[@rel=("mod","pc","ld") and ../@cat=("sv1","smain","ssub","inf","ppres","ppart","oti","ap","advp")]) 
         (: modification of verbal, adjectival heads :)
         (: nb some oti's directly dominate (preceding) modifiers :)
         (: [advp weg ermee ] :)
         then if ($node/../node[@rel=("hd","body") and (@pt or @cat)])  (: body for mods dangling outside cmp/body: maar niet om ... :)
              then local:label_vmod($node)
              else if ($node/../node[@rel=("su","obj1","predc","vc") and (@pt or @cat)])
                    then "orphan"
                    else if ($node[@rel="mod" and ../node[@rel=("pc","ld")]])
                          then "orphan"
                          else if ($node[@rel="mod" and ../node[@rel="mod"]/@begin < @begin]) (: gapping with multiple mods :)
                                then "orphan"
                                else local:dependency_label($node/..) (: gapping, where this mod is the head :)

    else if ($node[@rel="mod" and ../@cat=("pp","detp","advp")])
         then "amod"

    else if ($node[@rel="mod" and ../@cat=("cp", "whrel", "whq", "whsub")])
         (: [cp net  [cmp als] [body de Belgische regering]], net wat het land nodig heeft :)
         then if ($node/../node[@rel="body"]/node[@rel="hd" and @ud:pos="VERB"])
              then "advmod"
              else "amod"

    else if ($node[@rel="pc" and ../@cat="pp"])  (: [[hd,mwu aan het hoofd] [pc van X]] :)
     	  then "nmod"   
         
    else if ($node[@rel="hdf"])   then "case"
         
    else if ($node[@rel="predm"])
         then if ($node[@ud:pos]) 
               then "advmod"
               else "advcl"
         
    else if ( $node[@rel=("rhd","whd")] ) 
         then if ( $node/../node[@rel="body"]//node/number(@index) = $node/number(@index) ) 
              then local:non_local_dependency_label($node,($node/../node[@rel="body"]//node[number(@index) = $node/number(@index)])[1])
              else "advmod"  (: [whq waarom jij] :)                                                                             
       
    else if ($node[@rel="body"])
         then local:dependency_label($node/..)
         
    else if ($node[@rel="--"])
         then if ($node[@ud:pos="PUNCT"] )             
              then if ($node[not(../node[not(@ud:pos="PUNCT")]) and @begin = ../@begin]) then "root" (:just punctuation :)
              else "punct"   
         else if ($node[@ud:pos=("SYM","X") ] )               
              then if ($node/../node[@cat]) then "appos"  (: 1. Jantje is ziek  1-->appos?? :)
                   else "root"       
         else if ($node[@lemma="\\"] )                 then "punct"  (: hack for tagging errors in lassy small 250 :)
  (:       else if ($node[@spectype="deeleigen"] )       then "punct" :) (: hack for tagging errors in lassy small 250 :)   
         else if ($node[@ud:pos="NUM" and ../node[@cat] ] )  then "parataxis" (: dangling number 1. :)
         else if ($node[@ud:pos="CCONJ" and ../node[@cat="smain" or @cat="conj"]] ) then "cc" 
         (: sentence initial or final 'en' :)
         else if ($node[@ud:pos=("NOUN","PROPN","VERB") and ../node[@cat=("du","smain")]] ) then "parataxis" (: dangling words :)
         else if (count($node/../node[not(@ud:pos=("PUNCT","SYM","X"))]) < 2 ) then "root" (: only one non-punct/sym/foreign element in the string :)
         else if ($node[@cat="mwu"])
              then if ($node[@begin = ../@begin and @end = ../@end]) 
                   then "root"
                   else if ($node/node[@ud:pos=("PUNCT","SYM")]) (: fix for mwu punctuation in Alpino output :)
                        then  "punct"
                        else "ERROR_NO_LABEL_--"
         else if ($node[not(@ud:pos)]/../@rel="top")   then "root"
         else if ($node[@ud:pos="PROPN" and not(../node[@cat]) ] ) then "root"   (: Arthur . :)
         else if ($node[@ud:pos=("ADP","ADV","ADJ","DET","PRON","CCONJ","NOUN","VERB","INTJ")] )               then "parataxis"
         else "ERROR_NO_LABEL_--"
    
    else if ($node[@rel="hd"])
         then if ($node[@ud:pos="ADP"])
              then if ($node/../node[@rel="predc"]) 
                   then "mark" (: absolute met constructie -- analoog aan with X being Y :)
                   else if ( not($node/../node[@rel="pc"]) ) 
                         then "case"   (: er blijft weinig over van het lijk : over heads a predc and has pc as sister  :)
                         else local:dependency_label($node/..)  (: not sure about this one :)
         else if ($node[(@ud:pos=("ADJ","X","ADV") or @cat="mwu") 
         	             and ../@cat="pp" 
         	             and ../node[@rel=("obj1","se","vc")]]) 
                 then if ($node[../@rel="cnj" and ../node[@rel="obj1" and @index and not(@pt or @cat)]] )
                      then "conj"
                      else "case" (: vol vertrouwen in, Ad U3... , op het gebied van :)
         else if ($node/../node[@rel="hd"]/number(@begin) < $node/number(@begin) )
              then "conj"
              else local:dependency_label($node/..)
              
    else "ERROR_NO_LABEL"
};

(: this function is now also used to distribute dependents in coordination in Enhanced UD , so lot more rels and contexts are possible :)
declare function local:non_local_dependency_label($head as element(node), $gap as element(node)) as xs:string 
{ if      ($gap[@rel="su"])    
  then local:subject_label($gap)
  else if ($gap[@rel="obj1"])  
       then "obj"
  else if ($gap[@rel="obj2"])  
       then if ($head[@ud:pos="ADV"]) 
            then "advmod" 
            else "iobj"
  else if ($gap[@rel="me"  ])  
       then local:determine_nominal_mod_label($gap)
  else if ($gap[@rel=("predc","predm")]) 
       then local:dependency_label($gap)
  else if ($gap[@rel= ("pc", "ld")] )
       then if ($head/node[@rel="obj1"])
            then "nmod"
            else if ($head[@ud:pos=("ADV", "ADP") or @cat=("advp","ap")])    
                 then "advmod" (: waar precies zit je .. :)
                 else "ERROR_NO_LABEL_INDEX_PC"
  else if ($gap[@rel="pobj1"]) 
       then "expl"   (: waar het om gaat is dat hij scoort :) 
  else if ($gap[@rel="mwp"]) 
       then local:dependency_label($gap/..)   (: wat heb je voor boeken gelezen :)
  else if ($gap[@rel="vc"]) 
       then "ccomp"   (: wat ik me afvraag is of hij komt -- CLEFT:)
  else if ($gap[@rel="mod" and ../@cat="np"]) 
       then local:mod_label_inside_np($head)
  else if ($gap[@rel="mod" and ../@cat=("sv1","smain","ssub","inf","ppres","ppart","oti","ap","advp")])
       then local:mod_label_inside_np($head)
  else if ($gap[@rel="mod"]) 
       then if ($head[@cat=("pp","np") or @ud:pos=("NOUN","PRON")])
            then "nmod"
            else if ($head[@ud:pos=("ADV","ADP") or @cat= ("advp","ap","mwu","conj")]) 
                 then "advmod" (: hoe vaak -- AP, daar waar, waar en wanneer, voor als rhd :)
                 else "ERROR_NO_LABEL_INDEX_MOD"
  else if ($gap[@rel="det"]) 
       then local:det_label($head)
  else if ($gap[@rel="hd"] and $head[@ud:pos=("ADP","ADV")]) (: waaronder A, B, en C :)
       then "case"
  else if ($gap[@rel=("du","dp")]) 
       then "parataxis"
  else "ERROR_NO_LABEL_INDEX"
};

declare function local:label_vmod($node as element(node)) as xs:string {
 	if ($node[@cat="pp"]/node[@rel="vc"] ) then "advcl"
   	else if ($node[ (  node[@rel="hd" and @lemma="door"]
              	   	               or (@pt="bw" and ends-with(@lemma,"door"))
              	   	               )
              	   	               and ..[@cat="ppart"]/../node[@rel="hd" and @lemma=("worden","zijn")]
              	   	               and ../../node[@rel="su"]/@index = ../node[@rel="obj1"]/@index 
              	   	           ])
              	        then "obl:agent"  (: but NOT: door rookontwikkeling om het leven gekomen 
              	                             -- already filtered by constraint of su/obj1 control
              	                             NOT: bij Bakema is een stoeptegel door de ruit gegooid
              	                             NO/YES: hierdoor werd Prince door het grote publiek ontdekt :)
                   else if ($node[@cat=("pp","np","conj","mwu") or @ud:pos=("NOUN","PRON","PROPN","X","PUNCT","SYM") ]) then "obl"
                   else if ($node[@cat=("cp","sv1","smain","ppres","ppart","ti","oti","du","whq","whrel","rel")])  then "advcl"
                   else if ($node[@ud:pos= ("ADJ","ADV","ADP","VERB","SCONJ","INTJ") 
                   	              or @cat=("advp","ap")
                   	              or (@cat="conj" and node/@ud:pos="ADV")])  then "advmod"  (: niet of nauwelijks :)
                   else if ($node[@ud:pos="NUM"])    then "nummod"
                   else if ($node[@index])           then "ERROR_INDEX_VMOD"
                   else   "ERROR_NO_LABEL_VMOD"
};


declare function local:mod_label_inside_np($node as element(node)) as xs:string {
    if ($node[@cat="pp"]/node[@rel="vc"]) then "acl"  (: pp containing a clause :) 
    else if ($node[@ud:pos="ADJ" or @cat="ap" or node[@cat="conj" and node[@ud:POS="ADJ" or @cat="ap"] ]])      then "amod"
	else if ($node[@cat=("pp","np","conj","mwu") or @ud:pos=("NOUN","PRON","PROPN","X","PUNCT","SYM","INTJ") ]) then "nmod"
    else if ($node[@ud:pos="NUM"])             then "nummod"
    else if ($node[@cat="detp"])               then "det" (: [detp niet veel] meer error? :)
    else if ($node[@cat=("rel","whrel")])      then "acl:relcl"  
                (: v2 added relcl -- whrel= met name waar ... :)
    else if ($node[@cat="cp"]/node[@rel="body" and (@ud:pos = ("NOUN","PROPN") or @cat=("np","conj"))] ) then "nmod"   
                (: zijn loopbaan [CP als schrijver] :) 
    else if ($node[@cat=("cp","sv1","smain","ppres","ppart","ti","oti","du","whq") or @ud:pos="SCONJ"])  then "acl" 
                (: oa zinnen tussen haakjes :)
    else if ($node[@ud:pos= ("ADV","ADP","VERB","CCONJ") or @cat="advp"])  then "amod"
               (: VERB= aanstormend etc -> amod, ADV = nagenoeg alle prijzen, slechts 4 euro --> amod :)
               (: CCONJ = opdrachten zoals:   --> amod :)
    else if ($node[@index])     then "ERROR_INDEX_NMOD"
    else "ERROR_NO_LABEL_NMOD"
};

declare function local:det_label($node as element(node)) as xs:string 
{
 (: zijn boek, diens boek, ieders boek, aller landen, Ron's probleem, Fidel Castro's belang :)
  if (  $node[@ud:pos = "PRON" and @vwtype="bez"] or
        $node[@ud:pos = ("PRON","PROPN") and @naamval="gen"] or
        $node[@cat="mwu" and node[@spectype="deeleigen"]]
     ) 
  then "nmod:poss"
  else if ( $node/@ud:pos = ("DET","PROPN","NOUN","ADJ","PRON","ADV","X")  ) 
       then "det"   (: meer :)(: genoeg :) (:the :)
       else if ( $node/@cat = ("mwu","np","pp","ap","detp","smain") )                
            then "det" 
         (: tussen 5 en 6 .., needs more principled solution :)
         (: nog meer mensen dan anders  :)
         (: hetzelfde tijdstip als anders , nogal wat, :)
         (: hij heeft ik weet niet hoeveel boeken:)
            else if ( $node/@ud:pos = ("NUM","SYM") )           
                 then "nummod"
                 else if ( $node[@cat="conj"]) 
                      then if ($node/node[@rel="cnj"][1]/@ud:pos="NUM" )
                           then "nummod"
                           else "det"
                      else "ERROR_NO_LABEL_DET"
} ;

declare function local:determine_nominal_mod_label($node as element(node)) as xs:string
{ if ($node/../node[@rel="hd" and (@ud:pos="VERB" or @ud:pos="ADJ")]) 
  then "obl"
  else "nmod"
};

declare function local:determine_adjectival_mod_label($node as element(node)) as xs:string
{ if ($node/../node[@rel="hd" and (@ud:pos="VERB" or @ud:pos="ADJ")]) 
  then "obl"
  else "amod"
};

declare function local:subject_label($subj as element(node)) as xs:string 
{ let $cat :=
	if ( $subj[@cat=("whsub","ssub","ti","cp","oti")] or 
         $subj[@cat="conj" and node/@cat=("whsub","ssub","ti","cp","oti")]
	   ) 
    then "csubj" 
(: weather verbs and other expletive subject verbs :) 
    else if ($subj[@lemma="het"] and 
                ( $subj/../node[@rel="hd" and 
                                @lemma=("dooien", "gieten", "hagelen", "miezeren", 
                                        "misten", "motregenen", "onweren", "plenzen",
                                        "regenen", "sneeuwen", "stormen", "stortregenen",
                                        "ijzelen", "vriezen", "weerlichten", "winteren",
                                        "zomeren") ] or 
                  $subj/../node[@rel="hd" and 
                                @lemma="ontbreken" and 
                                ../node[@rel="pc" and node[@rel="hd" and @lemma="aan"] ] ] or
                  $subj[@index = ../node//node[@rel="sup"]/@index]    (: het kan voorkomen dat ... :) 
                )
             )
    then "expl" 
    else "nsubj"
  let $pass := local:passive_subject($subj) 
  return string-join(($cat,$pass),"")
};

declare function local:passive_subject($subj as element(node)) as xs:string 
{   if ( local:auxiliary(($subj/../node[@rel="hd"])[1]) eq "aux:pass" ) (: de carriere had gered kunnen worden :)
    then ":pass" 
    else if (local:auxiliary(($subj/../node[@rel="hd"])[1]) eq "aux"  and 
             $subj/@index = $subj/../node[@rel="vc"]/node[@rel="su"]/@index
            )
         then local:passive_subject($subj/../node[@rel="vc"]/node[@rel="su"])
         else ""
};

declare function local:conll-attribute($value as xs:string, $attribute as xs:string) as xs:string {
  if ($value eq 'null') 
  then "" 
  else string-join(($attribute,$value),"=")
};

declare function local:conll($node as element(node)) as xs:string*
{
for $word in $node//node[@word]

let $degree := if ($word/@ud:Degree)
               then local:conll-attribute($word/@ud:Degree,"Degree")
               else ""       
let $case   := if ($word/@ud:Case)
               then local:conll-attribute($word/@ud:Case,"Case")
               else ""
let $gender := if ($word/@ud:Gender)
               then local:conll-attribute($word/@ud:Gender,"Gender")
               else ""
let $person := if ($word/@ud:Person)
               then local:conll-attribute($word/@ud:Person,"Person")
               else ""
let $prontype := if ($word/@ud:PronType)
               then local:conll-attribute($word/@ud:PronType,"PronType")
               else ""
let $number := if ($word/@ud:Number)
               then local:conll-attribute($word/@ud:Number,"Number")
               else ""
let $reflex := if ($word/@ud:Reflex)
               then local:conll-attribute($word/@ud:Reflex,"Reflex")
               else ""
let $poss   := if ($word/@ud:Poss)
               then local:conll-attribute($word/@ud:Poss,"Poss")
               else ""
let $verbform := if ($word/@ud:VerbForm)
               then local:conll-attribute($word/@ud:VerbForm,"VerbForm")
               else ""
let $tense  := if ($word/@ud:Tense)
               then local:conll-attribute($word/@ud:Tense,"Tense")
               else ""
let $definite := if ($word/@ud:Definite)
               then local:conll-attribute($word/@ud:Definite,"Definite")
               else ""
let $foreign := if ($word/@ud:Foreign)
               then local:conll-attribute($word/@ud:Foreign,"Foreign")
               else ""
let $abbr := if ($word/@ud:Abbr)
               then local:conll-attribute($word/@ud:Abbr,"Abbr")
               else ""

              
               
let $features := replace(replace(replace(replace(
                               string-join(($abbr,$case,$definite,$degree,$foreign,$gender,$number,$person,$prontype,$reflex,$tense,$verbform),"|"),
                               "\|+","|"),
                               "^\|$","_"),
                               "^\|",""),
                               "\|$","")

let $quotes := $word/ancestor::node/descendant::node[@word=("'", '"')]/@begin

(: currently not used, but done in postprocessing by add_spaceafter.py:)
let $space_after := 
   if ($word/@end = 
   	     $word/ancestor::node/descendant::node[@word= (";",".",":","?","!",",",")","»")]/@begin)
               then "SpaceAfter=No"
               else if ($word/@word = ( "(" , "«" ) ) 
                    then "SpaceAfter=No"
                    else  if ($word/@word = ("'", '"')  and index-of($quotes,$word/@begin) mod 2 = 1 ) 
                          then "SpaceAfter=No"
                          else if ( $word/@end = 
   	                               $word/ancestor::node/descendant::node[@word= ("'",'"') and 
   	                                          index-of($quotes,@begin) mod 2 = 0]/@begin )
                                then "SpaceAfter=No"
                          else "_"

let $orig_postag := replace(replace(replace(replace($word/@postag,',','|'), '\(\)',''),'\(','|') ,'\)','')

order by number($word/@end)
return 
('&#10;',
string-join(($word/@end, $word/@word , $word/@lemma , $word/@ud:pos, $orig_postag, $features, $word/@ud:HeadPosition, $word/@ud:Relation, $word/@ud:Enhanced,"_"), "	" )
)
}; 


declare function local:sanity_check($node as element(node)) as element(node) {
let $count := count($node//node[@ud:Relation="root"])
let $zeroheadpos := count($node//node[@ud:HeadPosition="0"])
let $headpositionisself := count($node//node[@ud:HeadPosition=@end])
return
 element {name($node)} { ( $node/@*, 
                             attribute {"ud:roots"} {$count}, 
                             attribute {"ud:zeroheadpos"} {$zeroheadpos},
                             attribute {"ud:headpositionisself"} {$headpositionisself},
                             for $child in $node/node return $child ) }
};


for $href in doc($DIR)//doc/@href for $doc in doc($href)
    for $node in $doc/alpino_ds/node 
  return
  if ($MODE eq 'conll') 
  then <pre>
		<code sentence-id="{document-uri($doc)}">

  		{$node/../sentence,
              local:conll(local:enhanced_dependencies(
              	           local:add_dependency_relations(
              	             local:add_features(
              	             	local:add_pos_tags( 
              	             	   local:fix_misplaced_heads_in_coordination(
              	             	   	  local:fix_misplaced_heads_in_coordination($node)))))))}
          !
		  </code>
		</pre>
  else  <alpino_ds sentence-id="{document-uri($doc)}">
          { $node/../sentence,
            local:sanity_check(local:add_dependency_relations(
            	         local:add_features(
            	         	local:add_pos_tags( 
              	             	 (:)  local:fix_misplaced_heads_in_coordination( :)
              	             	   	  local:fix_misplaced_heads_in_coordination($node)))))
          }
        </alpino_ds>
  
