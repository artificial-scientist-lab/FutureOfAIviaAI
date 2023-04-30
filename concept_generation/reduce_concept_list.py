import pickle

full_concept_list = pickle.load( open( "full_concepts_20210320.pkl", "rb" ) )

nogo_word_list=['demonstrate','achieved','make','attracted','much','trained','improve','train','better','proposed','shown',
                'previous','achieves','every','quot','best','previously','important','becoming','become','brief','first',
                'visually','paper','present','poses','significant','learn','unknown','produce','multi','would','lead',
                'good','using','may','practical','last','year','recent','combined','resulting','published','allows','yields','debate',
                'derived','use','fit','within','associated','entire','autonomously','act','increasing','even','vast','simultaneously',
                'optimally','used','widely','studied','work','define','possibly','another','provides','beats',
                'encodes','among','made','regarding','needed','taken','based','performed','available','arises','applies',
                'includes','unless','uses','operates','sometimes','draws',
                'extends','lacks','various','tremendous','attains','scores','using','learns','generates','outperforms','compared','proposes','net','promising',
                'successfully','challenging','beyond','therefore','ral','certain','well','called','compared','towards','reproducing']

nogo_first_word_list=['art','article','tasks','attacks','sports','vis','dis','enforces','makes','functions','assumes','reads','pos','compares','fuses',
                      'effects','outputs','oblivious','bounds','languages','protects','indicates','laborious','offers','puts','robots','facilitates','measurements','arduous',
                      'proposals','similarities','assumptions','cameras','parameters','parsimonious','rs','variants','layers','labels','tests','serious','simulations','effectiveness','actions',
                      'problems','weights','givens','decades','analogous','determines','methods','arts','gans','experts','multipliers','outliers','issues','steps','contributions','nodes','patterns',
                      'benefits','challenges','times','parents','learns','assess','grows','minimizes','operations','ideas','stays','practitioners','decays','explanations','offs','mechanisms','techniques','guarantees','papers','companies','flops','goes','transfers','distills','always','populations','takes','items','numerous','observations','classes','leverages','leeds','agents','citypersons','cases','obtains','enormous','performs','pixels','obvious','stakes','aggregates','pomdps','domains','acts',
                      'algorithms','exploits','robotics','tedious','humans','users','scales','changes','boosts','customers','samples','strips','models','assists','predicts','extracts','excess','researchers','gives','address','chaos','seamless','requires','approaches','authors','representations','progress','receives','combines','becomes','depends','adds','success','contains','rewards','filters','insights','shows','policies','frameworks','embeddings','causes','finds','sheds','networks','words','sequences','communications','events','ups','maps','patches','things','converges','squares','surpass','systems','generates','correlates','captures','ones','points','destroys','builds','signals','ambitious','projects','generalizes','limitations','factors','options','conclusions','holds','pthreads','defocus','fields','keeps','perhaps','subspaces','markerless','ambiguous','magnitudes','connects','incorporates','process','hubness','pass','brings','years','prevents','utilizes','preserves','avoids','represents','roles','objectness','gets','assigns','indents','advantageous','balances','rays','hardness','across','produces','falls','less','asks','suppress','raises','reduces','remains','patients','synchronous','faces','engines','reaches','series','cnns','burgers','requirements','thus','occurs','variations','improves','reasons','exceeds','handles','homogeneous','difficulties','games','discuss','gradients','lossless','robustness','paths','examples','outperforms','tools','studies','descriptors','looks','encourages','integrates','demonstrates','computations','stokes','pays','diagnosis','edges','neurons','discontinuous','statistics','decides','enables','helps','aspects','classifiers','recommendations','advantages','suggests','differs','proofs','categories','es','maximizes','towards','lessons','datasets','gaussians','features''employs','projections','regions','artifacts','resources','countless','queries','graphs','tweets',
                      'surpasses','conditions','analyses','involves','leads','characterizes','matches','exhibits','iterates','increases','improvements','solutions','experiments','arms','besides','errors','programs','ts','results','applications','images','modes','saves','kitchens','moves','objects','means','variables','works','relies','semantics','instances','iterations','adapts','predictions','benchmarks','runs','findings','questions','quite','without','increased','interesting',
                      'simple','provide','easily','achieve','dimensional','optimize','jointly','tuned','primary','cannot','great','given','varied','generate','overall','seven','broadly','faster','old','past','handling','considered','often','highly','extracting','considerably','2nd','3rd','1st','thus','ultimately','substantial','realistic','current','learning','different','core','capture','level','efficient','aided','rapid','particularly','go','predict','generalize','improved','reduced','significantly','specific','inferior','existing','corresponding','discovering','multiple','suitable','known','directly']

nogo_last_word_list=['solves','manner','part','way','least','inter','must','high','obtains','reveals','exceeds','becomes','solves','comprises','operates','mi','via','extends','inter','utilizes','relies','pre','long','us','using','allows','learns','compared','u','one','lie','non','found','due','show','showing','using','according','changing','co','generalizes','real', 
                     'principle','data','automatic','developed']

nogo_words_with_rest=['second','higher']
nogo_words_this_rest=['order','order']

verbs_OK=['class','continuous','graphics','wireless','autonomous','cross','anomalous','videos','bioinformatics','spontaneous','fairness','boundedness','miss','simulations'
          'categories','jigsaws','bias','infectious','litis','restless','charades','smoothness','davis','closeness','hazardous','atlas','cityscapes','iris',
          'consensus','gross','smiles','ms','potts','mahalanobis','frobenius','modus','atrous','synthesis','gittins','isprs','correctness','los','kinetics','loss','fundus',
          'movielens','cmos','strips','heterogeneous','lanczos','hawkes','bsds','rts','brightness','seamless','cms','tsallis','bits','bus','rkhs','ais','nervous','porous','thesis','corpus','basis','sales','bayes',
          'gps','weightless','intraclass','matthews','focus','ddos','materials','diabetes','gas','noiseless','access','hypothesis','completeness','gibbs','mass','jones','medoids','squamous','sepsis','higgs','nuscenes','qrs','elites','liveness','cancerous','fitness','nas','hastings','chess','fictitious','textureless','business','radiomics','reynolds','sparseness','coronavirus','economics','nos','faithfulness','spurious','percutaneous','axis','physics','chervonenkis','interestingness','status','simultaneous','driverless','news','contiguous','mallows','multiclass','features','glass','employs','lens','las','pancreas','mathematics','constraints','lindenstrauss','malicious',
          'conscious','brats','lensless','harris','vicious','omics','exogenous','dexterous','ubiquitous','cs','inhomogeneous','census','tts','compress','mitosis','instantaneous','unconscious','analysis','massachusetts','dynamics','alzheimers','rigorous','asynchronous','osteoarthritis','suspicious']

nogo_first_word_ending_2letter=['ly']


len(full_concept_list)
new_reduced_list=[]
possible_verb_list=[]
possible_verb_list_indiv=[]
ccc=0
for one_concept in full_concept_list:
    one_concept=one_concept.replace('spiking neural networks spiking neural network','spiking neural network')

    separated_words=one_concept.split()

    do_remove=0
    for word in separated_words:
             
        # Remove numerics
        if word.isnumeric():
            do_remove=1
            
        if len(word)==1:
            do_remove=1
        
        # remove full nogo words
        for nogo_word in nogo_word_list:    
            if word==nogo_word:
#                if nogo_word==nogo_word_list[-1]:

                do_remove=1

        # remove with restriction on other words
        for ii in range(len(nogo_words_with_rest)):    
            if word==nogo_words_with_rest[ii] and (nogo_words_this_rest[ii] in separated_words)==False:
                do_remove=1                

    # remove if its first word
    if separated_words[0] in nogo_first_word_list:
        
        #if separated_words[0]==nogo_first_word_list[-1]:
        #    print(separated_words[0],': ',one_concept)        
        #    ccc+=0
        do_remove=1
        
    # remove if its first word
    tmp_first='    '+separated_words[0]
    if tmp_first[-2:] in nogo_first_word_ending_2letter:        
        print(one_concept)    
        do_remove=1        
        
    # remove if its last word
    if separated_words[-1] in nogo_last_word_list:
        
        if separated_words[-1]==nogo_last_word_list[-1]:
            #print(separated_words[-1],': ',one_concept)        
            ccc+=1
        do_remove=1        
    
    if do_remove==0:
        new_reduced_list.append(one_concept)
        
        if separated_words[0][-1]=='s' and not (separated_words[0] in verbs_OK):
            possible_verb_list.append(one_concept)
            possible_verb_list_indiv.append(separated_words[0])
        
print('ccc:',ccc)        
        
print(len(full_concept_list))
print(len(new_reduced_list))
         
print(len(new_reduced_list)/len(full_concept_list))
   
            
f = open("full_concepts_new.txt", "w")
for ii in range(len(new_reduced_list)):
    f.write(new_reduced_list[ii]+'\n')
f.close()


with open("full_concepts_new.pkl", "wb") as output_file:
    pickle.dump(new_reduced_list, output_file)