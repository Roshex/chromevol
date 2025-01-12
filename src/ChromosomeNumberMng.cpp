#include "ChromosomeNumberMng.h"
#include "ChromEvolOptions.h"

using namespace bpp;

void ChromosomeNumberMng::getCharacterData (const string& path){
    ChromosomeAlphabet* alphaInitial = new ChromosomeAlphabet(ChromEvolOptions::minAlpha_, ChromEvolOptions::maxAlpha_);
    VectorSequenceContainer* initialSetOfSequences = ChrFasta::readSequencesFromFile(path, alphaInitial);
    size_t numOfSequences = initialSetOfSequences->getNumberOfSequences();
    vector <string> sequenceNames = initialSetOfSequences->getSequenceNames();

    unsigned int maxNumberOfChr = 1; //the minimal number of chromosomes cannot be zero
    unsigned int minNumOfChr = ChromEvolOptions::maxAlpha_;

    std::vector <int> UniqueCharacterStates;
    cout<<"vector size is "<< UniqueCharacterStates.size()<<endl;
    for (size_t i = 0; i < numOfSequences; i++){
        BasicSequence seq = initialSetOfSequences->getSequence(sequenceNames[i]);
        int character = seq.getValue(0);
        if (character == -1){
            continue;
        }
        if (character == static_cast<int>(ChromEvolOptions::maxAlpha_)+1){
            continue;
        }
        // if it is a composite state
        if (character > static_cast<int>(ChromEvolOptions::maxAlpha_) +1){
            const std::vector<int> compositeCharacters = alphaInitial->getSetOfStatesForAComposite(character);
            for (size_t j = 0; j < compositeCharacters.size(); j++){
                if ((unsigned int) compositeCharacters[j] > maxNumberOfChr){
                    maxNumberOfChr = compositeCharacters[j];
                }
                if ((unsigned int) compositeCharacters[j] < minNumOfChr){
                    minNumOfChr = compositeCharacters[j];
                }
                
            }
            continue;
        }

        if (!std::count(UniqueCharacterStates.begin(), UniqueCharacterStates.end(), character)){
            UniqueCharacterStates.push_back(character);
        }
        if ((unsigned int) character > maxNumberOfChr){
            maxNumberOfChr = character;
        }
        if ((unsigned int) character < minNumOfChr){
            minNumOfChr = character;
        }

    }
    numberOfUniqueStates_ = (unsigned int)UniqueCharacterStates.size() + alphaInitial->getNumberOfCompositeStates();
    uint chrRangeNum = maxNumberOfChr - minNumOfChr;
    for (uint j = 1; j <= static_cast<uint>(ChromEvolOptions::numOfModels_); j++){      
        if (ChromEvolOptions::baseNum_[j] != IgnoreParam){
            if (ChromEvolOptions::baseNum_[j] > (int)chrRangeNum){
                chrRange_[j] = ChromEvolOptions::baseNum_[j] + 1;
            }else{
                chrRange_[j] = chrRangeNum;
            }
        }
    }
    cout <<"Number of unique states is " << numberOfUniqueStates_ <<endl;

    setMaxChrNum(maxNumberOfChr);
    setMinChrNum(minNumOfChr);

    vsc_ = resizeAlphabetForSequenceContainer(initialSetOfSequences, alphaInitial);
    delete initialSetOfSequences;
    delete alphaInitial;
    return;
}
/*************************************************************************************************************/
VectorSiteContainer* ChromosomeNumberMng::resizeAlphabetForSequenceContainer(VectorSequenceContainer* vsc, ChromosomeAlphabet* alphaInitial){
    size_t numOfSequences = vsc->getNumberOfSequences();
    vector <string> sequenceNames = vsc->getSequenceNames();
    alphabet_ = new ChromosomeAlphabet(ChromEvolOptions::minChrNum_,ChromEvolOptions::maxChrNum_);
        // fill with composite values
    if (alphaInitial->getNumberOfCompositeStates() > 0){
        const std::map <int, std::map<int, double>> compositeStates = alphaInitial->getCompositeStatesMap();
        std::map <int, std::map<int, double>>::const_iterator it = compositeStates.begin();
        while (it != compositeStates.end()){
            int compositeState = it->first;
            std::string charComposite = alphaInitial->intToChar(compositeState);
            alphabet_->setCompositeState(charComposite);
            it++;
        }
    }
    VectorSiteContainer* resized_alphabet_site_container = new VectorSiteContainer(alphabet_);
    for (size_t i = 0; i < numOfSequences; i++){
        BasicSequence seq = vsc->getSequence(sequenceNames[i]);
        BasicSequence new_seq = BasicSequence(seq.getName(), seq.getChar(0), alphabet_);
        resized_alphabet_site_container->addSequence(new_seq);

    }
    return resized_alphabet_site_container;
}


// /*******************************************************************************************************************/
std::map<std::string, std::map<string, double>> ChromosomeNumberMng::extract_alphabet_states(const string &file_path, int &min, int &max, vector<int> &uniqueStates, uint &numberOfComposite){
    
    ifstream stream;
    max = ChromEvolOptions::minAlpha_;
    min = ChromEvolOptions::maxAlpha_;
    std::regex rgx_composite("([\\d]+)=[\\d]+");
    std::regex rgx_prob("[\\d]+=([\\d]+\\.*[\\d]*)");
    std::regex rgx_state("([\\d]+)");
    std::regex rgx_species(">([\\S]+)");
    stream.open(file_path.c_str());
    vector <string> lines = FileTools::putStreamIntoVectorOfStrings(stream);
    stream.close();
    std::string species_name;
    std::map<std::string, std::map<string, double>> sp_with_states;
    for (size_t i = 0; i < lines.size(); i ++){
        vector <string> states;
        vector<double> probs;
        if (lines[i] == ""){
            continue;
        }else if (lines[i].rfind(">", 0) == 0){
            std::smatch match_sp_name;
            regex_search(lines[i], match_sp_name, rgx_species);
            species_name = match_sp_name[1];
            continue;
        }
        std::smatch match_composite;
        std::smatch match_prob;
        std::smatch match_single_state;
        string content = lines[i];
        bool composite = false;
        bool single_state = false;
        //uint state;
        while(regex_search(content, match_composite, rgx_composite))
        {
            composite = true;
            int state = stoi(match_composite[1]);
            if (state < min){
                min =  state;
            }
            if (state > max){
                max = state;
            }
            //state = static_cast<uint>(stoi(match[1]));
            
            states.push_back(match_composite[1]);

            regex_search(content, match_prob, rgx_prob);
            probs.push_back(std::stod(match_prob[1]));
            content = match_composite.suffix();
        }
        if (!composite){
            if (lines[i] == "X"){
                states.push_back("X");
                probs.push_back(1.0);
                single_state = true;
            }else if (regex_search(lines[i], match_single_state, rgx_state)){
                if ((size_t)(match_single_state[1].length()) == (size_t)(lines[i].length())){
                    //state = static_cast<uint>(stoi(match_single_state[1]));
                    int state = stoi(match_single_state[1]);
                    if (state < min){
                        min =  state;
                    }
                    if (state > max){
                        max = state;
                    }
                    states.push_back(match_single_state[1]);
                    single_state = true;
                    probs.push_back(1.0);
                    
                    if (!std::count(uniqueStates.begin(), uniqueStates.end(), state)){
                        uniqueStates.push_back(state);
                    }
                }
            }
            if (!single_state){
                throw Exception("Not a state!!!");
            }
        }else{
            numberOfComposite ++;
        }
        for (size_t j = 0; j < states.size(); j++){
            sp_with_states[species_name][states[j]] = probs[j];

        }

    }
    return sp_with_states;


}

/*******************************************************************************************/
void ChromosomeNumberMng::setMaxChrNum(unsigned int maxNumberOfChr){
    if (ChromEvolOptions::maxChrNum_ < 0){
        ChromEvolOptions::maxChrNum_ = maxNumberOfChr + std::abs(ChromEvolOptions::maxChrNum_);
    }else{
        if ((int)maxNumberOfChr > ChromEvolOptions::maxChrNum_){
            ChromEvolOptions::maxChrNum_ = maxNumberOfChr;
        }
    }

}
/****************************************************************************/
void ChromosomeNumberMng::setMinChrNum(unsigned int minNumberOfChr){
    if (ChromEvolOptions::minChrNum_ < 0){
        if (minNumberOfChr == 1){
            ChromEvolOptions::minChrNum_ = 0;
            std::cout << "Warning !!!! minChrNum_ should be at least 1!!" << std::endl;
            std::cout << "The mininal chromosome number was determined to be 1" << std::endl;
        }
        ChromEvolOptions::minChrNum_ = minNumberOfChr - std::abs(ChromEvolOptions::minChrNum_);
    }else{
        if ((int)minNumberOfChr < ChromEvolOptions::minChrNum_){
            ChromEvolOptions::minChrNum_ = minNumberOfChr;
        }

    }
}
/********************************************************************************************/

void ChromosomeNumberMng::getTree(const string& path, double treeLength){
    Newick reader;
    tree_ = reader.readPhyloTree(path);
    double treeLengthToScale = (treeLength > 0) ? treeLength : (double) numberOfUniqueStates_;
    rescale_tree(tree_, treeLengthToScale);
    return;

}
/****************************************************************************/
void ChromosomeNumberMng::rescale_tree(PhyloTree* tree, double chrRange){
    double scale_tree_factor = 1.0;
    bool rooted = tree->isRooted();
    if (!rooted){
        throw Exception("The given input tree is unrooted. Tree must be rooted!\n");
    }
    if (ChromEvolOptions::branchMul_ == 1.0){
        return;
    }else{
        //tree must be rescaled
        double treeLength = tree->getTotalLength();

        if ((ChromEvolOptions::branchMul_ == 999) || (ChromEvolOptions::treeLength_)){
            scale_tree_factor = chrRange/treeLength;
        }else{
            scale_tree_factor = ChromEvolOptions::branchMul_;
        }
        if (scale_tree_factor == 0){
            throw Exception("ChromosomeNumberMng::rescale_tree(): ERROR!!! Tree will be scaled to 0!!!!");
        }
        tree->scaleTree(scale_tree_factor);

    }

}
/*****************************************************************************************/
void ChromosomeNumberMng::getMaxParsimonyUpperBound(double* parsimonyBound) const{
    Newick reader;
    TreeTemplate<Node>* tree = reader.readTree(ChromEvolOptions::treeFilePath_);
    double factor = tree_->getTotalLength()/tree->getTotalLength();
    tree->scaleTree(factor);
    DRTreeParsimonyScore maxParsimonyObject = DRTreeParsimonyScore(*tree, *vsc_);
    *parsimonyBound = (maxParsimonyObject.getScore())/(tree->getTotalLength());
    delete tree;
    return;   

}

/*****************************************************************************************/
ChromosomeNumberOptimizer* ChromosomeNumberMng::optimizeLikelihoodMultiStartPoints() const{
    std::map<uint, std::pair<int, std::map<int, vector<double>>>> complexParamsValues;
    ChromEvolOptions::getInitialValuesForComplexParams(complexParamsValues);
    
    double parsimonyBound = 0;
    if (ChromEvolOptions::maxParsimonyBound_){
        getMaxParsimonyUpperBound(&parsimonyBound);
    }
    vector<uint> numOfIterationsForBackward = ChromEvolOptions::OptIterNumNextRounds_;
    vector<uint> numOfPointsForForward = ChromEvolOptions::OptPointsNumNextRounds_;
    if (!ChromEvolOptions::forwardPhase_){
        ChromEvolOptions::OptIterNumNextRounds_ = {0};
        ChromEvolOptions::OptPointsNumNextRounds_ = {1};

    }
    std::map<uint, uint> maxBaseNumTransition = (ChromEvolOptions::simulateData_ || ChromEvolOptions::useMaxBaseTransitonNumForOpt_) ? ChromEvolOptions::maxBaseNumTransition_ : chrRange_;
    ChromosomeNumberOptimizer* opt = new ChromosomeNumberOptimizer(tree_, alphabet_, vsc_, maxBaseNumTransition);
    //initialize all the optimization specific parameters
    opt->initOptimizer(ChromEvolOptions::OptPointsNum_, ChromEvolOptions::OptIterNum_, ChromEvolOptions::OptPointsNumNextRounds_, ChromEvolOptions::OptIterNumNextRounds_, ChromEvolOptions::optimizationMethod_, ChromEvolOptions::baseNumOptimizationMethod_,
        ChromEvolOptions::tolerance_, ChromEvolOptions::standardOptimization_, ChromEvolOptions::BrentBracketing_, 
        ChromEvolOptions::probsForMixedOptimization_);
    //optimize models
    time_t t1;
    time(&t1);
    time_t t2;
    std::map<uint, vector<uint>> mapOfNodesWithRoot = ChromEvolOptions::mapModelNodesIds_;
    mapOfNodesWithRoot[1].push_back(tree_->getRootIndex());
    auto modelAndRepresentitives = LikelihoodUtils::findMRCAForEachModelNodes(tree_, mapOfNodesWithRoot);
    opt->setInitialModelRepresentitives(modelAndRepresentitives);
    if (ChromEvolOptions::parallelization_){
        opt->optimizeInParallel(complexParamsValues, parsimonyBound, ChromEvolOptions::rateChangeType_, ChromEvolOptions::seed_, ChromEvolOptions::OptPointsNum_[0], ChromEvolOptions::fixedFrequenciesFilePath_, ChromEvolOptions::fixedParams_ ,ChromEvolOptions::mapModelNodesIds_);

    }else{
        opt->optimize(complexParamsValues, parsimonyBound, ChromEvolOptions::rateChangeType_, ChromEvolOptions::seed_, ChromEvolOptions::OptPointsNum_[0], ChromEvolOptions::fixedFrequenciesFilePath_, ChromEvolOptions::fixedParams_ ,ChromEvolOptions::mapModelNodesIds_);

    }
    ChromEvolOptions::OptIterNumNextRounds_ = numOfIterationsForBackward;
    ChromEvolOptions::OptPointsNumNextRounds_ = numOfPointsForForward;
    opt->setIterNumForNextRound(ChromEvolOptions::OptIterNumNextRounds_);
    opt->setPointsNumForNextRound(ChromEvolOptions::OptPointsNumNextRounds_);
    if (ChromEvolOptions::backwardPhase_){
        opt->optimizeBackwards(parsimonyBound, ChromEvolOptions::parallelization_);
    }


    time(&t2);
    std::cout <<"**** **** Total running time of the optimization procedure is: "<< (t2-t1) <<endl;
    return opt;
       
}
/******************************************************************************************************/
void ChromosomeNumberMng::getJointMLAncestralReconstruction(ChromosomeNumberOptimizer* optimizer, int* inferredRootState) const{
    vector<SingleProcessPhyloLikelihood*> vectorOfLikelihoods = optimizer->getVectorOfLikelihoods();
    // get the best likelihood
    SingleProcessPhyloLikelihood* lik = vectorOfLikelihoods[0];

    std::map<int, vector<pair<uint, int>>> sharedParams = optimizer->getSharedParams();
    uint numOfModels = static_cast<uint>(lik->getSubstitutionProcess().getNumberOfModels());
    std::map<int, std::map<uint, std::vector<string>>> typeWithParamNames;//parameter type, num of model, related parameters
    LikelihoodUtils::updateMapsOfParamTypesAndNames(typeWithParamNames, 0, lik, &sharedParams);
    std::map<uint, pair<int, std::map<int, std::vector<double>>>> modelsParams = LikelihoodUtils::getMapOfParamsForComplexModel(lik, typeWithParamNames, numOfModels);

    ParametrizablePhyloTree parTree = ParametrizablePhyloTree(*tree_);
    std::map<uint, std::vector<uint>> mapModelNodesIds;
    LikelihoodUtils::getMutableMapOfModelAndNodeIds(mapModelNodesIds, lik);
    std::map <uint, uint> baseNumberUpperBound;
    for (size_t m = 1; m <= numOfModels; m ++){
        auto branchProcess = lik->getSubstitutionProcess().getModel(m);
        baseNumberUpperBound[static_cast<uint>(m)] = std::dynamic_pointer_cast<const ChromosomeSubstitutionModel>(branchProcess)->getMaxChrRange();
    }
    std::shared_ptr<LikelihoodCalculationSingleProcess> likAncestralRec = setHeterogeneousLikInstance(lik, &parTree, baseNumberUpperBound, mapModelNodesIds, modelsParams, true);
    //auto likAncestralRec = std::make_shared<LikelihoodCalculationSingleProcess>(context, *sequenceData, *subProcess, rootFreqs);
    ParameterList paramsUpdated = likAncestralRec->getParameters();

    likAncestralRec->makeJointMLAncestralReconstruction();
    JointMLAncestralReconstruction* ancr = new JointMLAncestralReconstruction(likAncestralRec);
    ancr->init();
    std::map<uint, std::vector<size_t>> ancestors = ancr->getAllAncestralStates();
    std::map<uint, std::vector<size_t>>::iterator it = ancestors.begin();
    std::cout <<"******* ******* ANCESTRAL RECONSTRUCTION ******* ********" << endl;
    while(it != ancestors.end()){
        uint nodeId = it->first;
        if(!(tree_->isLeaf(tree_->getNode(nodeId)))){
            cout << "   ----> N-" << nodeId <<" states are: " << endl;
            for (size_t s = 0; s < ancestors[nodeId].size(); s++){
                cout << "           state: "<< ancestors[nodeId][s] + alphabet_->getMin() << endl;
                if (tree_->getRootIndex() == nodeId){
                    *inferredRootState = static_cast<int>(ancestors[nodeId][s]+ alphabet_->getMin());
                }
            }
        }else{
            cout << "   ----> " << (tree_->getNode(nodeId))->getName() << " states are: " << endl;
            for (size_t s = 0; s < ancestors[nodeId].size(); s++){
                cout << "           state: "<< ancestors[nodeId][s]+ alphabet_->getMin() << endl;

            }

        }
        it++;
    }
    const string outFilePath = ChromEvolOptions::resultsPathDir_ + "//" + "MLAncestralReconstruction.tree";
    PhyloTree* treeWithStates = tree_->clone();
    printTreeWithStates(*treeWithStates, ancestors, outFilePath);
    delete treeWithStates;


    delete ancr;
    //double likVal = likAncestralRec->makeJointMLAncestralReconstructionTest();
    std::cout << "********************************************\n";
    std::cout << " * * * * * * * * * * * * * * * * * * * * *\n";
    std::cout << "********************************************\n";
    auto sequenceData = likAncestralRec->getData();
    auto process = &(likAncestralRec->getSubstitutionProcess());
    auto context = &(likAncestralRec->getContext());
    delete process;
    delete sequenceData;
    delete context;
    
}
/***********************************************************************************/
std::map<int, vector<double>> ChromosomeNumberMng::getVectorToSetModelParams(SingleProcessPhyloLikelihood* lik, size_t modelIndex) const{
    
    ParameterList substitutionParams = lik->getSubstitutionModelParameters();
    std::map<int, vector <double>> compositeParams;

    for (size_t i = 0; i < ChromosomeSubstitutionModel::NUM_OF_CHR_PARAMS; i++){
        vector<string> paramNames;
        switch(i){
            case ChromosomeSubstitutionModel::BASENUM:   
                break;
            case ChromosomeSubstitutionModel::BASENUMR:
                paramNames = compositeParameter::getRelatedParameterNames(substitutionParams, "baseNumR");      
                break;
            case ChromosomeSubstitutionModel::DUPL:
                paramNames = compositeParameter::getRelatedParameterNames(substitutionParams, "dupl"); 
                break;
            case ChromosomeSubstitutionModel::LOSS:
                paramNames = compositeParameter::getRelatedParameterNames(substitutionParams, "loss");
                break;
            case ChromosomeSubstitutionModel::GAIN:
                paramNames = compositeParameter::getRelatedParameterNames(substitutionParams, "gain"); 
                break;
            case ChromosomeSubstitutionModel::DEMIDUPL:
                paramNames = compositeParameter::getRelatedParameterNames(substitutionParams, "demi"); 
                break;
            default:
                throw Exception("ChromosomeNumberMng::getVectorToSetModelParams(): Invalid rate type!");
                break;
        }
        if (i == ChromosomeSubstitutionModel::BASENUM){
            continue;
        }
        vector<double> paramValues;
        for (size_t j = 0; j < paramNames.size(); j++){
            paramValues.push_back(lik->getLikelihoodCalculationSingleProcess()->getParameter(paramNames[j]).getValue());
        }
        compositeParams[static_cast<int>(i)] = paramValues;


    }
    return compositeParams; 


}
/***********************************************************************************/
std::shared_ptr<NonHomogeneousSubstitutionProcess> ChromosomeNumberMng::setHeterogeneousModel(std::shared_ptr<ParametrizablePhyloTree> parTree, SingleProcessPhyloLikelihood* ntl, ValueRef <Eigen::RowVectorXd> rootFreqs,  std::map<int, vector<pair<uint, int>>> sharedParams) const{
    uint numOfModels = static_cast<uint>(ntl->getSubstitutionProcess().getNumberOfModels());
    std::map<int, std::map<uint, std::vector<string>>> typeWithParamNames;//parameter type, num of model, related parameters
    LikelihoodUtils::updateMapsOfParamTypesAndNames(typeWithParamNames, 0, ntl, &sharedParams);
    std::map<uint, pair<int, std::map<int, std::vector<double>>>> modelsParams = LikelihoodUtils::getMapOfParamsForComplexModel(ntl, typeWithParamNames, numOfModels);
    std::map<uint, std::vector<uint>> mapModelNodesIds;
    LikelihoodUtils::getMutableMapOfModelAndNodeIds(mapModelNodesIds, ntl);
    std::map <uint, uint> baseNumberUpperBound;
    for (size_t m = 1; m <= numOfModels; m ++){
        auto branchProcess = ntl->getSubstitutionProcess().getModel(m);
        baseNumberUpperBound[static_cast<uint>(m)] = std::dynamic_pointer_cast<const ChromosomeSubstitutionModel>(branchProcess)->getMaxChrRange();
    }
    auto rootFreqsValues =  rootFreqs->getTargetValue();
    Vdouble rootFreqsBpp;
    copyEigenToBpp(rootFreqsValues, rootFreqsBpp);
    std::shared_ptr<DiscreteDistribution> rdist = std::shared_ptr<DiscreteDistribution>(new GammaDiscreteRateDistribution(1, 1.0));
    
    std::shared_ptr<ChromosomeSubstitutionModel> chrModel = std::make_shared<ChromosomeSubstitutionModel>(alphabet_, modelsParams[1].second, modelsParams[1].first, baseNumberUpperBound[1], ChromosomeSubstitutionModel::rootFreqType::ROOT_LL, ChromEvolOptions::rateChangeType_);
    std::shared_ptr<FixedFrequencySet> rootFreqsFixed = std::make_shared<FixedFrequencySet>(std::shared_ptr<const StateMap>(new CanonicalStateMap(chrModel->getStateMap(), false)), rootFreqsBpp);
    std::shared_ptr<FrequencySet> rootFrequencies = std::shared_ptr<FrequencySet>(rootFreqsFixed->clone());
    std::shared_ptr<NonHomogeneousSubstitutionProcess> subProSim = std::make_shared<NonHomogeneousSubstitutionProcess>(rdist, parTree, rootFrequencies);

    // adding models
    for (uint i = 1; i <= numOfModels; i++){
        if (i > 1){
            chrModel = std::make_shared<ChromosomeSubstitutionModel>(alphabet_, modelsParams[i].second, modelsParams[i].first, baseNumberUpperBound[i], ChromosomeSubstitutionModel::rootFreqType::ROOT_LL, ChromEvolOptions::rateChangeType_);
        }   
        subProSim->addModel(std::shared_ptr<ChromosomeSubstitutionModel>(chrModel->clone()), mapModelNodesIds[i]);
    }
    return subProSim;


}
/***********************************************************************************/
std::shared_ptr<LikelihoodCalculationSingleProcess> ChromosomeNumberMng::setHeterogeneousLikInstance(SingleProcessPhyloLikelihood* likProcess, ParametrizablePhyloTree* tree, std::map<uint, uint> baseNumberUpperBound, std::map<uint, vector<uint>> &mapModelNodesIds, std::map<uint, pair<int, std::map<int, std::vector<double>>>> &modelParams, bool forAncestral) const{
    ValueRef <Eigen::RowVectorXd> rootFreqs = likProcess->getLikelihoodCalculationSingleProcess()->getRootFreqs();
    auto rootFreqsValues =  rootFreqs->getTargetValue();
    Vdouble rootFreqsBpp;
    copyEigenToBpp(rootFreqsValues, rootFreqsBpp);
    std::shared_ptr<DiscreteDistribution> rdist = std::shared_ptr<DiscreteDistribution>(new GammaDiscreteRateDistribution(1, 1.0));
    std::shared_ptr<ParametrizablePhyloTree> parTree = std::shared_ptr<ParametrizablePhyloTree>(tree->clone());
    
    uint numOfModels = static_cast<uint>(likProcess->getSubstitutionProcess().getNumberOfModels());
    std::shared_ptr<ChromosomeSubstitutionModel> chrModel = std::make_shared<ChromosomeSubstitutionModel>(alphabet_, modelParams[1].second, modelParams[1].first, baseNumberUpperBound[1], ChromosomeSubstitutionModel::rootFreqType::ROOT_LL, ChromEvolOptions::rateChangeType_);
    std::shared_ptr<FixedFrequencySet> rootFreqsFixed = std::make_shared<FixedFrequencySet>(std::shared_ptr<const StateMap>(new CanonicalStateMap(chrModel->getStateMap(), false)), rootFreqsBpp);
    //std::shared_ptr<FrequencySet> rootFrequencies = static_pointer_cast<FrequencySet>(rootFreqsFixed);
    std::shared_ptr<FrequencySet> rootFrequencies = std::shared_ptr<FrequencySet>(rootFreqsFixed->clone());
    std::shared_ptr<NonHomogeneousSubstitutionProcess> subProSim = std::make_shared<NonHomogeneousSubstitutionProcess>(rdist, parTree, rootFrequencies);

    // adding models
    for (uint i = 1; i <= numOfModels; i++){
        if (i > 1){
            chrModel = std::make_shared<ChromosomeSubstitutionModel>(alphabet_, modelParams[i].second, modelParams[i].first, baseNumberUpperBound[i], ChromosomeSubstitutionModel::rootFreqType::ROOT_LL, ChromEvolOptions::rateChangeType_);
        }   
        subProSim->addModel(std::shared_ptr<ChromosomeSubstitutionModel>(chrModel->clone()), mapModelNodesIds[i]);
    }


    SubstitutionProcess* nsubPro= subProSim->clone();
    Context* context = new Context();
    std::shared_ptr<LikelihoodCalculationSingleProcess> lik;
    if (forAncestral){
        lik = std::make_shared<LikelihoodCalculationSingleProcess>(*context, *vsc_->clone(), *nsubPro, rootFreqs);

    }else{
        lik = std::make_shared<LikelihoodCalculationSingleProcess>(*context, *vsc_->clone(), *nsubPro, false);
    }
    

    //delete subProSim;
    return lik;

}
/************************************************************************************/
// bool ChromosomeNumberMng::checkIfSimulationSuccess(string &simEvolutionPath){
//     bool success = true;
//     ifstream stream;
//     stream.open(simEvolutionPath.c_str());
//     // read entire file at once
//     const string content = string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
//     stream.close();
//     std::smatch match_from;
//     std::smatch match_to;
//     std::regex rgx_from("from state:[\\s]+([\\d]+)");
//     std::regex rgx_to("to state[\\s]+=[\\s]+([\\d]+)");
//     regex_search(content, match_from, rgx_from);
//     regex_search(content, match_to, rgx_to);
//     uint max_state = 0;
//     uint state;
//     string new_content = content;
//     while(regex_search(new_content, match_from, rgx_from))
//     {
//         state = static_cast<uint>(stoi(match_from[1]));
//         if (state > max_state){
//             max_state = state;
//         }
//         new_content = match_from.suffix();
//     }
//     new_content = content;
//     while(regex_search(new_content, match_to, rgx_to))
//     {
//         state = static_cast<uint>(stoi(match_to[1]));
//         if (state > max_state){
//             max_state = state;
//         }
//         new_content = match_to.suffix();
//     }
//     if (max_state == (uint)(ChromEvolOptions::maxChrNum_)){
//         success = false;
//     }
//     return success;


// }
/***********************************************************************************/
void ChromosomeNumberMng::simulateData(){
    RandomTools::setSeed(static_cast<long>(ChromEvolOptions::seed_));
    bool dataFileExists = false;
    if (FILE *file = fopen((ChromEvolOptions::characterFilePath_).c_str(), "r")) {
        fclose(file);
        dataFileExists = true;

    }
    bool simulateToDirs = false;
    bool dataFileIsDirectory = false;
    struct stat s;
    if ( lstat((ChromEvolOptions::characterFilePath_).c_str(), &s) == 0 ) {
        if (S_ISDIR(s.st_mode)) {
            dataFileIsDirectory = true;
        }
    }
    if ((dataFileExists) && (dataFileIsDirectory)){
        simulateToDirs = true;
            
    }
    if ((ChromEvolOptions::minChrNum_ <= 0) || (ChromEvolOptions::maxChrNum_ < 0)){
        throw Exception("ERROR!!! ChromosomeNumberMng::initializeSimulator(): minimum and maximum chromsome number should be positive!");
    }
    if (ChromEvolOptions::maxChrNum_ <= ChromEvolOptions::minChrNum_){
        throw Exception("ERROR!!! ChromosomeNumberMng::initializeSimulator(): maximum chromsome number should be larger than minimum chromosome number!");
    }
    alphabet_ = new ChromosomeAlphabet(ChromEvolOptions::minChrNum_, ChromEvolOptions::maxChrNum_);
    std::shared_ptr<DiscreteDistribution> rdist = std::shared_ptr<DiscreteDistribution>(new GammaDiscreteRateDistribution(1, 1.0));
    std::shared_ptr<ParametrizablePhyloTree> parTree =  std::make_shared<ParametrizablePhyloTree>(*tree_);
    std::map<uint, std::pair<int, std::map<int, vector<double>>>> complexParamsValues;
    ChromEvolOptions::getInitialValuesForComplexParams(complexParamsValues);
    std::map<uint, uint> maxBaseNumTransition = (ChromEvolOptions::simulateData_) ? ChromEvolOptions::maxBaseNumTransition_ : chrRange_;
    //1. ChromEvolOptions::mapModelNodesIds_: already calculated
    
    std::shared_ptr<ChromosomeSubstitutionModel> chrModel = std::make_shared<ChromosomeSubstitutionModel>(alphabet_, complexParamsValues[1].second, complexParamsValues[1].first, maxBaseNumTransition[1], ChromosomeSubstitutionModel::rootFreqType::ROOT_LL, ChromEvolOptions::rateChangeType_, true);
    if ((chrModel->getBaseNumber() != IgnoreParam) && (ChromEvolOptions::correctBaseNumber_)){
        chrModel->correctBaseNumForSimulation(ChromEvolOptions::maxChrInferred_);

    }
    
    if (ChromEvolOptions::fixedFrequenciesFilePath_ == "none"){
        throw Exception("ChromosomeNumberMng::initializeSimulator(): ERROR! The file of fixed root frequencies is missing!!!");  

    }
    vector <double> rootFreqs = LikelihoodUtils::setFixedRootFrequencies(ChromEvolOptions::fixedFrequenciesFilePath_, chrModel);
    FrequencySet* rootFreqsFixed = new FixedFrequencySet(std::shared_ptr<const StateMap>(new CanonicalStateMap(chrModel->getStateMap(), false)), rootFreqs);
    std::shared_ptr<FrequencySet> rootFrequencies = std::shared_ptr<FrequencySet>(rootFreqsFixed->clone());
    std::shared_ptr<NonHomogeneousSubstitutionProcess> subProSim = std::make_shared<NonHomogeneousSubstitutionProcess>(rdist, parTree, rootFrequencies);

    // adding models
    for (uint i = 1; i <= (uint)(ChromEvolOptions::numOfModels_); i++){
        if (i > 1){
            chrModel = std::make_shared<ChromosomeSubstitutionModel>(alphabet_, complexParamsValues[i].second, complexParamsValues[i].first, maxBaseNumTransition[i], ChromosomeSubstitutionModel::rootFreqType::ROOT_LL, ChromEvolOptions::rateChangeType_, true);
            if (chrModel->getBaseNumber() != IgnoreParam){
                chrModel->correctBaseNumForSimulation(ChromEvolOptions::maxChrInferred_);

            }
            
        }   
        subProSim->addModel(chrModel, ChromEvolOptions::mapModelNodesIds_[i]);
    }
    SimpleSubstitutionProcessSiteSimulator* simulator = new SimpleSubstitutionProcessSiteSimulator(*subProSim);
    size_t counter = 0;
    for (size_t i = 0; i < ChromEvolOptions::numOfSimulatedData_; i++){
        if (simulateToDirs){
            string simDirPath = ChromEvolOptions::resultsPathDir_ +"//"+ std::to_string(i);
            if (FILE *file = fopen(simDirPath.c_str(), "r")) {
                fclose(file);

            }else{
                if (mkdir(simDirPath.c_str(), 0700) == -1){
                    throw Exception("Directory was not created!!!");
                }

            }

        }
            
        simulateData(simulateToDirs, i, counter, simulator);
        if ((double)counter > (double)(ChromEvolOptions::fracAllowedFailedSimulations_)*(double)(ChromEvolOptions::numOfSimulatedData_)){
            throw Exception("ChromosomeNumberMng::runChromEvol():Too many failed simulations!");
            return;
        }else{
            if ((i+1)-counter == ChromEvolOptions::numOfRequiredSimulatedData_){
                std::cout << "Found " << ChromEvolOptions::numOfRequiredSimulatedData_ << " successful simulations" << std::endl;
                break;

            }
        }

    }
    delete simulator;
           
    return;

}

/***********************************************************************************/
void ChromosomeNumberMng::runChromEvol(){
    LikelihoodUtils::setNodeIdsForAllModels(tree_, ChromEvolOptions::mapModelNodesIds_, ChromEvolOptions::nodeIdsFilePath_, ChromEvolOptions::initialModelNodes_);
    if (ChromEvolOptions::simulateData_){
        time_t t1;
        time(&t1);
        time_t t2;
        //simulate data using a tree and a set of model parameters  
        simulateData();
        time(&t2);
        std::cout <<"**** **** Total running time of the simulation procedure is: "<< (t2-t1) <<endl;
        return;

    }
    
    // optimize likelihood
    ChromosomeNumberOptimizer* chrOptimizer = optimizeLikelihoodMultiStartPoints();
    ////////////////////////////////////////////////////////////////
    // !!!!! Note !!!!! The first model should be the root model!!!
    ////////////////////////////////////////////////////////////////
    
    // get joint ML ancestral reconstruction
    int inferredRootState;
    getJointMLAncestralReconstruction(chrOptimizer, &inferredRootState);
    writeOutputToFile(chrOptimizer, inferredRootState);
    //get Marginal ML ancestral reconstruction, and with the help of them- calculate expectations of transitions
    const string outFilePath = ChromEvolOptions::resultsPathDir_ +"//"+ "ancestorsProbs.txt";
    getMarginalAncestralReconstruction(chrOptimizer, outFilePath);
    // test stochastic mapping
    if (ChromEvolOptions::runStochasticMapping_){
        runStochasticMapping(chrOptimizer);

    }
    //compute expectations
    computeExpectations(chrOptimizer, ChromEvolOptions::NumOfSimulations_);
    //The optimizer is deleted inside the computeExpectations object!



}
/**************************************************************************************/
void ChromosomeNumberMng::runStochasticMapping(ChromosomeNumberOptimizer* chrOptimizer){

    auto likObject = chrOptimizer->getVectorOfLikelihoods()[0];
    bool weightedRootFreqs = (ChromEvolOptions::fixedFrequenciesFilePath_ != "none") ? true:false;
    auto lik = likObject->getLikelihoodCalculationSingleProcess();  
    std::map<int, vector<pair<uint, int>>> sharedParams = chrOptimizer->getSharedParams();
    ParametrizablePhyloTree tree =  ParametrizablePhyloTree(*tree_);
    std::shared_ptr<ParametrizablePhyloTree> parTree = std::shared_ptr<ParametrizablePhyloTree>((&tree)->clone());
    ValueRef <Eigen::RowVectorXd> rootFreqs = likObject->getLikelihoodCalculationSingleProcess()->getRootFreqs();
    std::shared_ptr<NonHomogeneousSubstitutionProcess> multiModelProcess = setHeterogeneousModel(parTree, likObject, rootFreqs, sharedParams);
    SubstitutionProcess* nsubPro= multiModelProcess->clone();
    Context context;
    auto likObjectOpt = std::make_shared<LikelihoodCalculationSingleProcess>(context, *vsc_->clone(), *nsubPro, weightedRootFreqs);
    auto likTest = SingleProcessPhyloLikelihood(context, likObjectOpt, likObjectOpt->getParameters());
    std::cout << "stochastic mapping lik object likelihood: " << likTest.getValue();

    StochasticMapping* stm = new StochasticMapping(likObjectOpt, ChromEvolOptions::NumOfSimulations_, ChromEvolOptions::numOfStochasticMappingTrials_);//ChromEvolOptions::NumOfSimulations_);
    stm->generateStochasticMapping();
    // retry to get unsuccessful nodes via stretching the problematic branches.
    // initalizing a map that will contain all the branches that were stretched with their original lengths.
    std::map<uint, std::map<size_t, std::pair<double, double>>> originalBrLenVsStretched;
    StochasticMappingUtils::fixFailedMappings(tree_, stm, originalBrLenVsStretched, ChromEvolOptions::numOfFixingMappingIterations_);

    // getting expected number of transitions for each type (comparable to the expectation computation).
    // This is just a test!!
    std::map<uint, std::map<pair<size_t, size_t>, double>> expectationsPerNode = stm->getExpectedNumOfOcuurencesForEachTransitionPerNode();
    auto nonHomoProcess = dynamic_cast<const NonHomogeneousSubstitutionProcess*>(&(likObjectOpt->getSubstitutionProcess()));
    std::map<int, double> expectationsTotal = ComputeChromosomeTransitionsExp::getExpectationsPerType(nonHomoProcess, *tree_, expectationsPerNode);
    std::cout << "*** *** *** Test stochastic mapping *** *** ***:" << std::endl;
    auto it = expectationsTotal.begin();
    while (it != expectationsTotal.end()){
        std::cout << it->first << ": " << expectationsTotal[it->first] << std::endl;
        it ++;
    }
    // get the expected rates for each transition
    std::map<uint, std::map<size_t, bool>> presentMapping;
    auto rootToLeafTransitions = stm->getNumOfOccurrencesFromRootToNode(presentMapping);
    //const string outStMappingRootToLeaf = ChromEvolOptions::resultsPathDir_+"//"+ "stMapping_root_to_leaf.txt";
    StochasticMappingUtils::printRootToLeaf(tree_, alphabet_, rootToLeafTransitions, presentMapping, ChromEvolOptions::NumOfSimulations_, nonHomoProcess, ChromEvolOptions::resultsPathDir_);
    
    Vdouble dwellingTimesPerState = stm->getDwellingTimesUnderEachState();
    auto numOfOccurencesPerTransition = stm->getTotalNumOfOcuurencesForEachTransition();
    VVdouble ratesPerTransition = stm->getExpectedRateOfTransitionGivenState(dwellingTimesPerState, numOfOccurencesPerTransition);
    const string outStMappingPath = ChromEvolOptions::resultsPathDir_+"//"+ "stochastic_mapping.txt";
    // print all the results associated with stochastic mapping
    printStochasticMappingResults(stm, dwellingTimesPerState, numOfOccurencesPerTransition, ratesPerTransition, expectationsTotal, outStMappingPath);
    size_t numOfMapping = stm->getNumberOfMappings();
    const std::shared_ptr<PhyloTree> stmTree = stm->getTree();
    auto &mappings = stm->getMappings();
    auto &ancestralStates = stm->getAncestralStates();
    const string stretchedBranchesPath = ChromEvolOptions::resultsPathDir_+"//"+ "stretchedBranchesForMapping.csv";
    StochasticMappingUtils::printMappingStretchedBranches(stretchedBranchesPath, originalBrLenVsStretched, stmTree);

    for (size_t i = 0; i < numOfMapping; i++){
        const string outPathPerMapping =  ChromEvolOptions::resultsPathDir_+"//"+ "evoPathMapping_" + std::to_string(i) + ".txt";
        StochasticMappingUtils::printStochasticMappingEvolutionaryPath(alphabet_, stmTree, mappings, ancestralStates, i, outPathPerMapping);
    }

    // delete
    auto sequenceData = likObjectOpt->getData();
    auto process = &(likObjectOpt->getSubstitutionProcess());
    delete process;
    delete sequenceData;
    delete stm;
    


}
/**************************************************************************************/
void ChromosomeNumberMng::printStochasticMappingResults(StochasticMapping* stm, Vdouble &dwellingTimesPerState, std::map<pair<size_t, size_t>, double> &numOfOccurencesPerTransition, VVdouble &ratesPerTransition, std::map<int, double> &expectationsTotal, const string &outStMappingPath){       
    ofstream outFileStMapping;
    outFileStMapping.open(outStMappingPath);
    outFileStMapping << "*** *** Expected rates *** ***" << std::endl;
    for (size_t beginState = 0; beginState < ratesPerTransition.size(); beginState++){
        for (size_t endState = 0; endState < ratesPerTransition[beginState].size(); endState++){
            if (ratesPerTransition[beginState][endState] == 0){
                continue;
            }
            outFileStMapping << "\t" << beginState + alphabet_->getMin() << " -> " << endState + alphabet_->getMin() << ": " << ratesPerTransition[beginState][endState] << std::endl;
        

        }
    }
    outFileStMapping << "*** *** Total duration times *** ***" << std::endl;
    for (size_t beginState = 0; beginState < dwellingTimesPerState.size(); beginState++){
        outFileStMapping << "\t" << beginState + alphabet_->getMin() << ": " << dwellingTimesPerState[beginState] << std::endl;

    }
    outFileStMapping << "*** *** Total number of occurrences *** ***" << std::endl;
    auto itTransitions = numOfOccurencesPerTransition.begin();
    while(itTransitions != numOfOccurencesPerTransition.end()){
        auto start = (itTransitions->first).first;
        auto end = (itTransitions->first).second;
        outFileStMapping << "\t" << start + alphabet_->getMin() << " -> " << end + alphabet_->getMin() << ": " << numOfOccurencesPerTransition[itTransitions->first] << std::endl;
        itTransitions ++;
    }
    stm->printUnrepresentedLeavesWithCorrespondingMappings(outFileStMapping);
    outFileStMapping << "*** *** Expected number of transitions per type *** ***" << std::endl;
    for (int p = 0; p < ChromosomeSubstitutionModel::typeOfTransition::NUMTYPES; p++){
        if (p == ChromosomeSubstitutionModel::GAIN_T){
            outFileStMapping <<"\tGAIN: " <<  expectationsTotal[p] << std::endl;
        }else if (p == ChromosomeSubstitutionModel::LOSS_T){
            outFileStMapping <<"\tLOSS: " <<  expectationsTotal[p] << std::endl;
        }else if (p == ChromosomeSubstitutionModel::DUPL_T){
            outFileStMapping <<"\tDUPLICATION: " <<  expectationsTotal[p] << std::endl;
        }else if (p == ChromosomeSubstitutionModel::DEMIDUPL_T){
            outFileStMapping <<"\tDEMI-DUPLICATION: " <<  expectationsTotal[p] << std::endl;
        }else if (p == ChromosomeSubstitutionModel::BASENUM_T){
            outFileStMapping <<"\tBASE-NUMBER: " <<  expectationsTotal[p] << std::endl;
        }else if (p == ChromosomeSubstitutionModel::MAXCHR_T){
            outFileStMapping <<"\tTOMAX: " <<  expectationsTotal[p] << std::endl;
        }
    }

    outFileStMapping.close();

}
/**************************************************************************************/
void ChromosomeNumberMng::getMarginalAncestralReconstruction(ChromosomeNumberOptimizer* chrOptimizer, const string &filePath){
    vector<SingleProcessPhyloLikelihood*> vectorOfLikelihoods = chrOptimizer->getVectorOfLikelihoods();
    // get the best likelihood
    SingleProcessPhyloLikelihood* lik = vectorOfLikelihoods[0];
    auto singleLikProcess = lik->getLikelihoodCalculationSingleProcess();
    vector<shared_ptr<PhyloNode> > nodes = tree_->getAllNodes();
    size_t nbNodes = nodes.size();
    MarginalAncestralReconstruction *asr = new MarginalAncestralReconstruction(singleLikProcess);
    std::map<uint, VVdouble> posteriorProbs;
    std::map<uint, vector<size_t>> mapOfAncestors;
    for (size_t n = 0; n < nbNodes; n++){
        uint nodeId = tree_->getNodeIndex(nodes[n]);
        posteriorProbs[nodeId].reserve(1);//one site
        mapOfAncestors[nodeId] = asr->getAncestralStatesForNode(nodeId, posteriorProbs[nodeId], false); 
    }
    ofstream outFile;
    outFile.open(filePath);
    outFile << "NODE";
    for (size_t i = 0; i < alphabet_->getSize(); i ++){
        outFile << "\t" << (i + alphabet_->getMin());
    }
    outFile <<"\n";
    std::map<uint, std::vector<size_t>>::iterator it = mapOfAncestors.begin();
    while(it != mapOfAncestors.end()){
        uint nodeId = it->first;
        if(!(tree_->isLeaf(tree_->getNode(nodeId)))){
            outFile << "N-" << nodeId;
        }else{
            outFile << (tree_->getNode(nodeId))->getName();
        }
        for (size_t i = 0; i < posteriorProbs[nodeId][0].size(); i ++){
            outFile << "\t" << (posteriorProbs[nodeId][0][i]);

        }
        outFile << "\n";

        it++;
    }

    outFile.close();
    const string outFilePath = ChromEvolOptions::resultsPathDir_ +"//"+"MarginalAncestralReconstruction.tree";
    printTreeWithStates(*tree_, mapOfAncestors, outFilePath);
    delete asr;
}

/**************************************************************************************/
void ChromosomeNumberMng::printSimulatedEvoPath(const string outPath, SiteSimulationResult* simResult, bool &success, size_t maxStateIndex) const{
    ofstream outFile;
    success = true;
    outFile.open(outPath);
    size_t totalNumTransitions = 0;
    vector<shared_ptr<PhyloNode> > nodes = tree_->getAllNodes();
    size_t nbNodes = nodes.size();
    for (size_t n = 0; n < nbNodes; n++){
        uint nodeId = tree_->getNodeIndex(nodes[n]);
        if (tree_->getRootIndex() == nodeId){
            outFile << "N-" + std::to_string(nodeId) << endl;
            size_t rootState = simResult->getRootAncestralState();
            if (rootState == maxStateIndex){
                success = false;
            }
            outFile <<"\tThe root state is: "<< ((int)(rootState + alphabet_->getMin())) <<endl;


        }else{
            if (tree_->isLeaf(nodeId)){
                outFile << tree_->getNode(nodeId)->getName() << endl;
            }else{
                outFile << "N-" + std::to_string(nodeId) <<endl;

            }
            MutationPath mutPath = simResult->getMutationPath(nodeId);
            vector<size_t> states = mutPath.getStates();
            vector<double> times = mutPath.getTimes();
            totalNumTransitions += static_cast<int>(times.size());

            auto edgeIndex =  tree_->getIncomingEdges(nodeId)[0]; 
            auto fatherIndex = tree_->getFatherOfEdge(edgeIndex);
            outFile << "Father is: " << "N-" << fatherIndex << std::endl;
            size_t fatherState;
            if (fatherIndex == tree_->getRootIndex()){
                fatherState = simResult->getRootAncestralState() + alphabet_->getMin();    
            }else{
                fatherState = simResult->getAncestralState(fatherIndex) + alphabet_->getMin(); 
            }
            for (size_t i = 0; i < states.size(); i++){
                double time;
                if (i == 0){
                    time = times[i];
                }else{
                    time =  times[i]-times[i-1];
                }
                outFile << "from state: "<< fatherState  <<"\tt = "<<time << " to state = "<< ((int)(states[i]) + alphabet_->getMin()) << endl;
                if (((size_t)(fatherState-alphabet_->getMin()) == maxStateIndex) || (states[i] == maxStateIndex)){
                    success = false;
                }
                fatherState = ((int)(states[i]) + alphabet_->getMin());
            }
            outFile <<"# Number of transitions per branch: "<< times.size() <<endl;   
            
        }
        
        outFile <<"*************************************"<<endl;
        
    }
    outFile <<"Total number of transitions is: "<< totalNumTransitions << endl;
    outFile.close();

}

void ChromosomeNumberMng::printTreeWithStates(PhyloTree tree, std::map<uint, std::vector<size_t>> &ancestors, const string &filePath) const{
    uint rootId = tree.getRootIndex();
    convertNodesNames(tree, rootId, ancestors);
    string tree_str = printTree(tree);
    cout << tree_str << endl;
    if (filePath != "none"){
       ofstream outFile;
       outFile.open(filePath);
       outFile << tree_str << endl;
       outFile.close();
    }

}
/**************************************************************************************/
void ChromosomeNumberMng::convertNodesNames(PhyloTree &tree, uint nodeId, std::map<uint, std::vector<size_t>> &ancestors, bool alphabetStates) const{
    size_t state = ancestors[nodeId][0];
    if (alphabetStates){
        state += alphabet_->getMin();
    }
    
    if (tree.isLeaf(nodeId)){
        string prevName = tree.getNode(nodeId)->getName();
        const string newName = (prevName + "-"+ std::to_string(state));
        tree.getNode(nodeId)->setName(newName);

    }else{
        // internal node -> N[nodeId]-[state]
        string prevName = "N" + std::to_string(nodeId);
        const string newName = (prevName + "-"+ std::to_string(state));
        tree.getNode(nodeId)->setName(newName);
        auto sons = tree.getSons(tree.getNode(nodeId));
        //auto sons = tree.getNode(nodeId)->getSons();
        for (size_t i = 0; i < sons.size(); i++){
            uint sonId = tree.getNodeIndex(sons[i]);
            convertNodesNames(tree, sonId, ancestors, alphabetStates);

        }
    }
}


/****************************************************************************************/
string ChromosomeNumberMng::printTree(const PhyloTree& tree)
{
  ostringstream s;
  s << "(";
  uint rootId = tree.getRootIndex();
  auto node = tree.getNode(rootId);
  if (tree.isLeaf(rootId) && node->hasName()) // In case we have a tree like ((A:1.0)); where the root node is an unamed leaf!
  {
    s << node->getName();
    auto sons = tree.getSons(node);
    for (size_t i = 0; i < sons.size(); ++i)
    {
        uint sonId = tree.getNodeIndex(sons[i]);
        s << "," << nodeToParenthesis(sonId, tree);
    }
  }
  else
  {
    auto sons = tree.getSons(node) ;
    uint firstSonId = tree.getNodeIndex(sons[0]);
    s << nodeToParenthesis(firstSonId, tree);
    for (size_t i = 1; i < sons.size(); ++i)
    {
        uint sonId = tree.getNodeIndex(sons[i]);
        s << "," << nodeToParenthesis(sonId, tree);
    }
  }
  s << ")";
  s << tree.getNode(rootId)->getName();

  s << ";" << endl;
  return s.str();
}

/******************************************************************************/
string ChromosomeNumberMng::nodeToParenthesis(const uint nodeId, const PhyloTree &tree)
{
  ostringstream s;
  if (tree.isLeaf(tree.getNode(nodeId)))
  {
    s << tree.getNode(nodeId)->getName();
  }
  else
  {
    s << "(";
  
    auto sons = tree.getSons(tree.getNode(nodeId));
    uint firstSonId = tree.getNodeIndex(sons[0]);
    s << nodeToParenthesis(firstSonId, tree);
    for (size_t i = 1; i < sons.size(); i++)
    {
        uint sonId = tree.getNodeIndex(sons[i]);
        
        s << "," << nodeToParenthesis(sonId, tree);
    }
    s << ")";
  }
  if (!tree.isLeaf(tree.getNode(nodeId))){
      s << tree.getNode(nodeId)->getName();
  }
  shared_ptr<PhyloBranch> branch=tree.getEdgeToFather(nodeId);
  if (branch->hasLength()){
    s << ":" << branch->getLength();

  }

  return s.str();
}


/*********************************************************************************/
void ChromosomeNumberMng::simulateData(bool into_dirs, size_t simNum, size_t &count_failed, SimpleSubstitutionProcessSiteSimulator* simulator){
    SiteSimulationResult* simResult = simulator->dSimulateSite();
    vector <size_t> leavesStates = simResult->getFinalStates();
    vector<string> leavesNames = simResult->getLeaveNames();
    string countsPath;
    string evolutionPath;
    string ancestorsPath;
    if (into_dirs){
        countsPath = ChromEvolOptions::resultsPathDir_ +"//"+ std::to_string(simNum) + "//"+ "counts.fasta";
        ancestorsPath = ChromEvolOptions::resultsPathDir_ +"//"+ std::to_string(simNum) +"//"+"simulatedDataAncestors.tree";
        evolutionPath = ChromEvolOptions::resultsPathDir_ +"//"+ std::to_string(simNum) +"//"+ "simulatedEvolutionPaths.txt";
    }else{
        countsPath = ChromEvolOptions::characterFilePath_;
        ancestorsPath = ChromEvolOptions::resultsPathDir_  +"//"+"simulatedDataAncestors.tree";
        evolutionPath = ChromEvolOptions::resultsPathDir_ +"//"+"simulatedEvolutionPaths.txt";
    }
    printSimulatedData(leavesStates, leavesNames, 0, countsPath);
    printSimulatedDataAndAncestors(simResult, ancestorsPath);
    if (ChromEvolOptions::resultsPathDir_ != "none"){
        bool success;
        size_t maxStateIndex = (size_t)(ChromEvolOptions::maxChrNum_-alphabet_->getMin());
        printSimulatedEvoPath(evolutionPath, simResult, success, maxStateIndex);
        
        if (!success){
            count_failed ++;

        }
    }
    delete simResult;

}
/*******************************************************************************/
void ChromosomeNumberMng::printSimulatedData(vector<size_t> leavesStates, vector<string> leavesNames, size_t iter, string &countsPath){
    cout << "Simulated data #" << iter << endl;
    for (size_t i = 0; i < leavesNames.size(); i++){
        cout << leavesNames[i] << " "<< leavesStates[i] + alphabet_->getMin() <<endl;
    }
    cout << "******************************"<<endl;
    
    if (ChromEvolOptions::resultsPathDir_ != "none"){
        //create vector site container object and save fasta file.
        VectorSiteContainer* simulatedData = new VectorSiteContainer(alphabet_);
        for (size_t i = 0; i < leavesNames.size(); i++){
            int state = (int)leavesStates[i] + alphabet_->getMin();
            BasicSequence seq = BasicSequence(leavesNames[i], alphabet_->intToChar(state), static_cast <const Alphabet*>(alphabet_));
            simulatedData->addSequence(seq);
        }
        vsc_ = simulatedData;
        
        Fasta fasta;
        fasta.writeSequences(countsPath, *simulatedData);

    }


    
}
/****************************************************************************/
void ChromosomeNumberMng::printSimulatedDataAndAncestors(SiteSimulationResult* simResult, string &ancestorsPath) const{
    std::map<uint, std::vector<size_t> > ancestors;
    vector<shared_ptr<PhyloNode> > nodes = tree_->getAllNodes();
    size_t nbNodes = nodes.size();
    for (size_t n = 0; n < nbNodes; n++){
        uint nodeId = tree_->getNodeIndex(nodes[n]);
        vector<size_t> nodesStates;
        if (nodeId == tree_->getRootIndex()){
            nodesStates.push_back(simResult->getRootAncestralState()); 
        }else{
            nodesStates.push_back(simResult->getAncestralState(nodeId));
        }       
        ancestors[nodeId] = nodesStates;
    }
    if (ChromEvolOptions::resultsPathDir_ == "none"){
        printTreeWithStates(*tree_, ancestors, ChromEvolOptions::resultsPathDir_);
    }else{
        printTreeWithStates(*tree_, ancestors, ancestorsPath);
    }
  
}

/*****************************************************************************************************/

void ChromosomeNumberMng::computeExpectations(ChromosomeNumberOptimizer* chrOptimizer, int numOfSimulations) const{
    if (numOfSimulations == 0){
        return;
    }
    std::cout << "Strating the computation of expectations ...." << std::endl;
    vector<SingleProcessPhyloLikelihood*> vectorOfLikelihoods = chrOptimizer->getVectorOfLikelihoods();
    // get the best likelihood
    SingleProcessPhyloLikelihood* ntl = vectorOfLikelihoods[0];
    auto lik = ntl->getLikelihoodCalculationSingleProcess();
    
    std::map<int, vector<pair<uint, int>>> sharedParams = chrOptimizer->getSharedParams();
    ParametrizablePhyloTree* tree =  new ParametrizablePhyloTree(*tree_);
    std::shared_ptr<ParametrizablePhyloTree> parTree = std::shared_ptr<ParametrizablePhyloTree>(tree);
    ValueRef <Eigen::RowVectorXd> rootFreqs = ntl->getLikelihoodCalculationSingleProcess()->getRootFreqs();
    std::shared_ptr<NonHomogeneousSubstitutionProcess> multiModelProcess =  setHeterogeneousModel(parTree, ntl, rootFreqs, sharedParams);


    size_t nbStates = alphabet_->getSize();
    std::map <uint, std::map<size_t, VVdouble>> jointProbabilitiesFatherSon;
    uint rootId = tree_->getRootIndex();
    vector<shared_ptr<PhyloNode> > nodes = tree_->getAllNodes();
    size_t nbNodes = nodes.size();
    for (size_t n = 0; n < nbNodes; n++){
        uint nodeId = tree_->getNodeIndex(nodes[n]);
        if (nodeId == rootId){
            continue;
        }
        jointProbabilitiesFatherSon[nodeId][0].reserve(nbStates);
        lik->makeJointLikelihoodFatherNode_(nodeId, jointProbabilitiesFatherSon[nodeId][0], 0, 0);
      
    }
    std::cout << "Finished with the calculation of joint likelihoods of father and son..."<< std::endl;
    std::cout << "Starting running simulations ... " << std::endl;

    delete chrOptimizer;
    //initializing the expectation instance
    ComputeChromosomeTransitionsExp* expCalculator = new ComputeChromosomeTransitionsExp(multiModelProcess, tree_, alphabet_, jointProbabilitiesFatherSon, ChromEvolOptions::jumpTypeMethod_);
    expCalculator->runSimulations(numOfSimulations);
    std::cout << "Simulations are done. Now strating with the conputation of expectation per type..." << std::endl;
    expCalculator->computeExpectationPerType();
    std::cout << "Computation is done ... Printing results ... " << std::endl;
    if (ChromEvolOptions::resultsPathDir_ == "none"){
        expCalculator->printResults();
    }else{
        const string outFilePath = ChromEvolOptions::resultsPathDir_+"//"+ "expectations.txt";
        //const string outFilePathForNonAccounted = ChromEvolOptions::resultsPathDir_+"//"+ "exp_nonAccounted_branches.txt";
        const string outFilePathHeuristics = ChromEvolOptions::resultsPathDir_+"//"+ "expectations_second_round.txt";
        const string outTreePath = ChromEvolOptions::resultsPathDir_+"//"+ "exp.tree";
        expCalculator->printResults(outFilePath);
        std::cout << "Run heuristics if needed ..." << std::endl;
        expCalculator->runHeuristics();
        std::cout << "Printing heuristics results ... " << std::endl;
        expCalculator->printResults(outFilePathHeuristics);
        std::cout << "Printing the tree of expectations ... " << std::endl;
        PhyloTree* expTree = expCalculator->getResultTree();
        string tree_str = printTree(*expTree);
        std::cout << "Done! Deleting all unneeded objects!" << std::endl;
        delete expTree;
        ofstream treeFile;
        treeFile.open(outTreePath);
        treeFile << tree_str;
        treeFile.close();
    }
    

    delete expCalculator;
}
/******************************************************************************/
void ChromosomeNumberMng::writeOutputToFile(ChromosomeNumberOptimizer* chrOptimizer, int &inferredRootState) const{
    double AICc = chrOptimizer->getAICOfBestModel();
    auto bestLik = chrOptimizer->getVectorOfLikelihoods()[0];
    std::map<uint, std::vector<uint>> mapModelNodesIds;
    LikelihoodUtils::getMutableMapOfModelAndNodeIds(mapModelNodesIds, bestLik, tree_->getRootIndex());

    const string outPath = (ChromEvolOptions::resultsPathDir_ == "none") ? (ChromEvolOptions::resultsPathDir_) : (ChromEvolOptions::resultsPathDir_ + "//" + "chromEvol.res");
    ofstream outFile;
    if (outPath != "none"){
        outFile.open(outPath);
    }
    outFile << "#################################################" << std::endl;
    outFile << "Running parameters" << std::endl;
    outFile << "#################################################" << std::endl;
    writeRunningParameters(outFile);
    outFile << "#################################################" << std::endl;
    outFile << "Best chosen model" <<std::endl;
    outFile << "#################################################" << std::endl;


    auto numOfModels = bestLik->getSubstitutionProcess().getNumberOfModels();
    outFile << "Number of models in the best model = " << numOfModels << std::endl;
    // not all the assignements of the nodes induce clades, therefore the min clade size will represent the 
    // number of species under a specific model (better ask Itay)
    uint minSizeOfClade = findMinCladeSize(mapModelNodesIds);
    outFile << "Min clade size in the best model = " << minSizeOfClade << std::endl;
    outFile << "Root node is: " << "N" << tree_->getRootIndex() << std::endl;
    outFile << "Ancestral chromosome number at the root: " << inferredRootState <<std::endl;
    auto modelAndRepresentitives = LikelihoodUtils::findMRCAForEachModelNodes(tree_, mapModelNodesIds);
    outFile << "Shifting nodes are: " << std::endl;
    for (uint i = 1; i <= numOfModels; i++){
        for (size_t j = 0; j < modelAndRepresentitives[i].size(); j++){
            if (modelAndRepresentitives[i][j] == tree_->getRootIndex()){
                outFile << "# Model $" << i << " = " << "N" << modelAndRepresentitives[i][j] << " (the root)" << std::endl;
            }else{
                outFile << "# Model $" << i << " = " << "N" << modelAndRepresentitives[i][j] << std::endl;
            }        
        }
        
    }
    writeTreeWithCorrespondingModels(*tree_, mapModelNodesIds);

    chrOptimizer->printRootFrequencies(bestLik, outFile);
    printLikParameters(chrOptimizer, bestLik, outFile);
    outFile << ChromEvolOptions::modelSelectionCriterion_ <<" of the best model = "<< AICc << std::endl;
    auto previousModelsPartitions = chrOptimizer->getPreviousModelsPartitions();
    auto previousModelsAICcAndLik = chrOptimizer->getPreviousModelsAICcValues();
    auto previousModelsParameters = chrOptimizer->getPreviousModelsParameters();
    auto previousRootFrequencies = chrOptimizer->getPrevModelsRootFreqs();
    uint initialNumOfModels = ChromEvolOptions::numOfModels_;
    if (!previousModelsAICcAndLik.empty()){
        outFile << "#################################################" << std::endl;
        outFile << "Previous best chosen models in the forward phase" <<std::endl;
        outFile << "#################################################" << std::endl;
        for (size_t i = 0; i < previousModelsParameters.size(); i++){
            uint model = (uint)i + initialNumOfModels;
            outFile << "*** Number of shifts: " << model-1 << " ***" << std::endl;
            outFile << "Shifting nodes are:" << std::endl;
            for (size_t j = 0; j < previousModelsPartitions[model].size(); j++){
                if (tree_->getRootIndex() == previousModelsPartitions[model][j]){
                    outFile << "\tModel $" << j + 1 << " = " << "N" << previousModelsPartitions[model][j] << " (the root)" << std::endl;
                }else{
                    outFile << "\tModel $" << j + 1 << " = " << "N" << previousModelsPartitions[model][j] << std::endl;

                }
            }
            outFile << "Ancestral probabilities at the root are:" << std::endl;
            outFile << "\t";
            auto rootFreqs = previousRootFrequencies[model];
            for (size_t j= 0; j < rootFreqs.size(); j++){
                if (j == rootFreqs.size() -1){
                    outFile << "F[" << j + alphabet_->getMin() << "] = " << rootFreqs[j] << endl;
                }else{
                    outFile << "F[" << j + alphabet_->getMin() << "] = " << rootFreqs[j] << "\t";

                }   
            }
            outFile << "Model parameters are:" << std::endl;
            for (size_t j = 0; j < (previousModelsParameters[model]).size(); j++){
                 outFile << "\t" << (previousModelsParameters[model])[j].first << " = " << (previousModelsParameters[model])[j].second << std::endl;
            }
            outFile << ChromEvolOptions::modelSelectionCriterion_ << " of the best model with " << model-1 << " shifts = " << (previousModelsAICcAndLik[model]).first << std::endl;
            outFile << "Log likelihood = " << (previousModelsAICcAndLik[model]).second << std::endl;
            outFile << "\n";
            
        }


    }


    outFile.close();

}
void ChromosomeNumberMng::writeRunningParameters(ofstream &outFile) const{
    outFile << "Min allowed chromosome number  = " << alphabet_->getMin() << std::endl;
    outFile << "Max allowed chromosome number = " << alphabet_->getMax() << std::endl;
    outFile <<"Number of tips in the tree = " << tree_->getAllLeavesNames().size() << std::endl;
    auto originalTreeLength = getOriginalTreeLength(ChromEvolOptions::treeFilePath_);
    outFile << "Initial number of models was set to : " << ChromEvolOptions::numOfModels_ << std::endl;
    outFile << "Max number of models was set to ";
    if ((ChromEvolOptions::heterogeneousModel_) && (ChromEvolOptions::maxNumOfModels_ == 1)){
        outFile << "be inferred in the forward phase (i.e., indefinite)" << std::endl;
    }else{
       outFile << ChromEvolOptions::maxNumOfModels_ << std::endl; 
    }
    outFile << "The inferred model was set to be : ";
    if (ChromEvolOptions::heterogeneousModel_){
        outFile << "heterogeneous" << std::endl;
    }else{
        outFile << "homogeneous" << std::endl;
    }
    outFile << "_branchMul was set to: " << ChromEvolOptions::branchMul_ << std::endl;
    outFile << "_treeLength was set to: " << ChromEvolOptions::treeLength_ << std::endl;
    outFile << "Original tree length was: " << originalTreeLength <<std::endl;
    outFile << "Tree scaling factor is: " << tree_->getTotalLength()/originalTreeLength << std::endl;
    outFile << "tree Length was scaled to: " << tree_->getTotalLength() << std::endl;
    outFile << "Min clade size specified in the parameter file = " << ChromEvolOptions::minCladeSize_ << std::endl;
    outFile << "_OptPointsNum was set to: ";
    for (size_t i = 0; i < ChromEvolOptions::OptPointsNum_.size(); i++){
        if (i != ChromEvolOptions::OptPointsNum_.size()-1){
            outFile << ChromEvolOptions::OptPointsNum_[i]  << ", "; 
        }else{
            outFile << ChromEvolOptions::OptPointsNum_[i]  <<std::endl; 
        }
    }
    outFile << "_OptIterNum was set to: ";
    for (size_t i = 0; i < ChromEvolOptions::OptIterNum_.size(); i++){
        if (i != ChromEvolOptions::OptIterNum_.size()-1){
            outFile << ChromEvolOptions::OptIterNum_[i]  << ", "; 
        }else{
            outFile << ChromEvolOptions::OptIterNum_[i]  <<std::endl; 
        }
    }
    if (ChromEvolOptions::heterogeneousModel_){
        outFile << "_OptPointsNumNextRounds was set to: ";
        for (size_t i = 0; i < ChromEvolOptions::OptPointsNumNextRounds_.size(); i++){
            if (i != ChromEvolOptions::OptPointsNumNextRounds_.size()-1){
                outFile << ChromEvolOptions::OptPointsNumNextRounds_[i]  << ", "; 
            }else{
                outFile << ChromEvolOptions::OptPointsNumNextRounds_[i]  <<std::endl; 
            }
        }
        outFile << "_OptIterNumNextRounds was set to: ";
        for (size_t i = 0; i < ChromEvolOptions::OptIterNumNextRounds_.size(); i++){
            if (i != ChromEvolOptions::OptIterNumNextRounds_.size()-1){
                outFile << ChromEvolOptions::OptIterNumNextRounds_[i]  << ", "; 
            }else{
                outFile << ChromEvolOptions::OptIterNumNextRounds_[i]  <<std::endl; 
            }
        }
        std::cout << "delta " << ChromEvolOptions::modelSelectionCriterion_ << " threshold was set to: " << ChromEvolOptions::deltaAICcThreshold_ << std::endl;

    }
    outFile << "Optimization method was set to: " << ChromEvolOptions::optimizationMethod_ << std::endl;
    outFile << "_probsForMixedOptimization was set to: ";
    for (size_t i = 0; i < ChromEvolOptions::probsForMixedOptimization_.size(); i++){
        if (i == ChromEvolOptions::probsForMixedOptimization_.size()-1){
            outFile << ChromEvolOptions::probsForMixedOptimization_[i] << " for gradient descent" << std::endl;
        }else{
            outFile << ChromEvolOptions::probsForMixedOptimization_[i] << " for Brent "<< ", ";
        }
    }
    outFile << "Seed was set to: " << ChromEvolOptions::seed_ << std::endl;
    outFile << "Base number optimization method as set to: " << ChromEvolOptions::baseNumOptimizationMethod_ << std::endl;
    outFile << "Root frequencies were set to: " << ChromEvolOptions::rootFreqs_ << std::endl;
    outFile << "_maxParsimonyBound was set to: ";
    if (ChromEvolOptions::maxParsimonyBound_){
        outFile << "true" << std::endl; 
    }else{
        outFile << "false" << std::endl;
    }
    outFile << "_simulateData was set to: ";
    if (ChromEvolOptions::simulateData_){
        outFile << "true" << std::endl; 
    }else{
        outFile << "false" << std::endl;
    }
    if (ChromEvolOptions::simulateData_){
        outFile << "_maxBaseNumTransition set to:" << std::endl;
        for (uint m = 1; m <= (uint)ChromEvolOptions::numOfModels_; m++){
            outFile << "\t" << "Model #" << m << ": " << ChromEvolOptions::maxBaseNumTransition_[m] << std::endl;
        }
        outFile << "_maxChrInferred is set to: " << ChromEvolOptions::maxChrInferred_ << std::endl;
    }
    outFile << "Number of simulations for the expectation computation is set to: " << ChromEvolOptions::NumOfSimulations_ << std::endl;
    outFile << "_parallelization was set to: ";
    if (ChromEvolOptions::parallelization_){
        outFile << "true" << std::endl;
    }else{
        outFile << "false" << std::endl;
    }
    outFile << "Initial parameters were set to:" << std::endl;
    for (uint i = 1; i <= (uint)ChromEvolOptions::numOfModels_; i++){
        for (size_t j = 0; j < ChromEvolOptions::gain_[i].size(); j++){
            outFile <<"\tChromosome.gain" << j << "_" << i << " = " << ChromEvolOptions::gain_[i][j] << std::endl;
        }
        for (size_t j = 0; j < ChromEvolOptions::loss_[i].size(); j++){
            outFile <<"\tChromosome.loss" << j << "_" << i << " = " << ChromEvolOptions::loss_[i][j] << std::endl;
        }
        for (size_t j = 0; j < ChromEvolOptions::dupl_[i].size(); j++){
            outFile <<"\tChromosome.dupl" << j << "_" << i << " = " << ChromEvolOptions::dupl_[i][j] << std::endl;
        }
        for (size_t j = 0; j < ChromEvolOptions::demiDupl_[i].size(); j++){
            outFile <<"\tChromosome.demi" << j << "_" << i << " = " << ChromEvolOptions::demiDupl_[i][j] << std::endl;
        }
        for (size_t j = 0; j < ChromEvolOptions::baseNumR_[i].size(); j++){
            outFile <<"\tChromosome.baseNumR" << j << "_" << i << " = " << ChromEvolOptions::baseNumR_[i][j] << std::endl;
        }
        outFile << "\tChromosome.baseNum_" <<i << " = " << ChromEvolOptions::baseNum_[i] << std::endl;
    }
    outFile << "Assigned functions for each rate parameter:" << std::endl;
    size_t startForComposite = ChromosomeSubstitutionModel::getNumberOfNonCompositeParams();
    for (int i = (int)startForComposite; i < ChromosomeSubstitutionModel::paramType::NUM_OF_CHR_PARAMS; i++){
        string paramName = LikelihoodUtils::getStringParamName(i);
        string functionName = LikelihoodUtils::getFunctionName(ChromEvolOptions::rateChangeType_[(size_t)i-startForComposite]);
        outFile << "\t" << paramName << ": " << functionName << std::endl;
    }
    if (ChromEvolOptions::heterogeneousModel_){
        outFile << "Global parameters that were set to be equal accross different models:" << std::endl;
        for (size_t i = 0; i < ChromEvolOptions::globalParams_.size(); i++){
            outFile << "\t" << ChromEvolOptions::globalParams_[i] << std::endl;

        }
        if (ChromEvolOptions::globalParams_.size() == 0){
            outFile << "\tNo global parameters were defined."  << std::endl;
        }
    }
    outFile << "Shared parameters that were set to be equal within each model:" << std:: endl;
    auto it = ChromEvolOptions::sharedParameters_.begin();
    size_t numOfShared = 0;
    while (it != ChromEvolOptions::sharedParameters_.end()){
        auto sharedParams = ChromEvolOptions::sharedParameters_[it->first];
        if (sharedParams.size() < 2){
            continue;
            it ++;
        }
        numOfShared += sharedParams.size();
        for (size_t j = 0; j < sharedParams.size(); j++){

            if (j == sharedParams.size()-1){
                outFile << "\t" << LikelihoodUtils::getStringParamName(sharedParams[j].second) << "_" << sharedParams[j].first << std::endl;
            }else{
                outFile << "\t" << LikelihoodUtils::getStringParamName(sharedParams[j].second) << "_" << sharedParams[j].first << " = ";
            }
        }

        it ++;
    }
    if (numOfShared == 0){
        outFile << "\tNo shared parameters were defined." << std::endl;
    }
    outFile << "Fixed parameters were set to:" << std::endl;
    if(ChromEvolOptions::fixedParams_.size() > 0){
        for (uint i = 1; i <= (uint)ChromEvolOptions::numOfModels_; i++){
            if (ChromEvolOptions::fixedParams_.find(i) != ChromEvolOptions::fixedParams_.end()){
                if (ChromEvolOptions::fixedParams_[i].size() == 0){
                    outFile << "\tModel #" << i << ": No fixed parameters were set." << std::endl;
                }else{
                    for (size_t j = 0; j < ChromEvolOptions::fixedParams_[i].size(); j++){
                        outFile << "\tModel #" << i << LikelihoodUtils::getStringParamName(ChromEvolOptions::fixedParams_[i][j]) << std::endl;

                    }   
                }
            }
        }      
    }else{
        outFile << "\tNo fixed parameters were defined." << std::endl;
    }

}

void ChromosomeNumberMng::writeTreeWithCorrespondingModels(PhyloTree tree, std::map<uint, vector<uint>> &modelAndNodes) const{
    std::map<uint, std::vector<size_t>> mapOfNodeAndModel;
    auto it = modelAndNodes.begin();
    while (it != modelAndNodes.end()){
        size_t model = (size_t)(it->first);
        for (size_t i = 0; i < modelAndNodes[it->first].size(); i++){
            uint nodeId = modelAndNodes[it->first][i];
            mapOfNodeAndModel[nodeId].push_back(model);
        }
        it ++;
    }
    uint rootId = tree.getRootIndex();
    convertNodesNames(tree, rootId, mapOfNodeAndModel, false);
    string tree_str = printTree(tree);
    //outFile << tree_str << std::endl;
    string pathForTree = ChromEvolOptions::resultsPathDir_ +"//"+ "treeWithShifts.tree";
    ofstream outFileTree;
    outFileTree.open(pathForTree);
    outFileTree << tree_str << std::endl;
    outFileTree.close();

}

/******************************************************************************/
uint ChromosomeNumberMng::findMinCladeSize(std::map<uint, vector<uint>> mapModelNodesIds) const{
    auto it = mapModelNodesIds.begin();
    uint minNumSpecies = (uint)(tree_->getAllLeavesNames().size());
    if (mapModelNodesIds.size() == 1){
        return minNumSpecies;
    }
    while (it != mapModelNodesIds.end()){
        auto nodes = mapModelNodesIds[it->first];
        uint numOfSpecies = 0;
        for (size_t i = 0; i < nodes.size(); i++){
            uint nodeId = nodes[i];
            if (tree_->isLeaf(nodeId)){
                numOfSpecies ++;
            }
        }
        if (minNumSpecies > numOfSpecies){
            minNumSpecies = numOfSpecies;
        }

        it ++;
    }
    return minNumSpecies;
}
/******************************************************************************/
void ChromosomeNumberMng::printLikParameters(ChromosomeNumberOptimizer* chrOptimizer, SingleProcessPhyloLikelihood* lik, ofstream &outFile) const{

    outFile << "Final optimized likelihood is: "<< lik->getValue() << endl;
    outFile << "Final model parameters are:"<<endl;
    ParameterList substitutionModelParams = lik->getSubstitutionModelParameters();
    size_t numOfModels = lik->getSubstitutionProcess().getNumberOfModels();
    std::vector<std::string> paramsNames = substitutionModelParams.getParameterNames();
    std::map<uint, std::map<int, std::vector<std::string>>> mapOfTypeAndName;
    std::map<pair<uint, int>, vector<string>> mapOfAliasedTypeModelAndParam;

    auto sharedParams = chrOptimizer->getSharedParams();
    auto it  = sharedParams.begin();
    while(it != sharedParams.end()){
        // get the string full names of the first parameter to which the other ones are aliased
        auto sharedParametersBlock = sharedParams[it->first];
        uint firstParamModel = sharedParametersBlock[0].first;
        int firstType = sharedParametersBlock[0].second;
        uint numOfSubParams = LikelihoodUtils::getNumberOfParametersPerParamType(firstType, ChromEvolOptions::rateChangeType_);      
        string basicName = LikelihoodUtils::getStringParamName(firstType);
        std::vector<string> firstParamNames;
        if (firstType == ChromosomeSubstitutionModel::BASENUM){
            firstParamNames.push_back("Chromosome." + basicName + "_"+ std::to_string(firstParamModel));
        }else{
            for (size_t i = 0; i < numOfSubParams; i++){
                firstParamNames.push_back("Chromosome." + basicName +std::to_string(i)+"_"+ std::to_string(firstParamModel));
            }

        }
        // add the aliased parameters to the map, such that they will correspond to the name of the first parameter to which they are aliased
        for (size_t i = 0; i < sharedParametersBlock.size(); i++){
            pair<uint, int> paramModelAndType;
            paramModelAndType.first = sharedParametersBlock[i].first;
            paramModelAndType.second = sharedParametersBlock[i].second;
            string shortName = LikelihoodUtils::getStringParamName(sharedParametersBlock[i].second);
            if (paramModelAndType.second == ChromosomeSubstitutionModel::BASENUM){
                mapOfAliasedTypeModelAndParam[paramModelAndType].push_back("Chromosome." + shortName + "_"+ std::to_string(paramModelAndType.first));
                mapOfTypeAndName[sharedParametersBlock[i].first][sharedParametersBlock[i].second].push_back(firstParamNames[0]);
            }else{
                for (size_t j = 0; j < numOfSubParams; j++){
                    mapOfAliasedTypeModelAndParam[paramModelAndType].push_back("Chromosome." + shortName +std::to_string(j)+ "_"+ std::to_string(paramModelAndType.first));
                    mapOfTypeAndName[sharedParametersBlock[i].first][sharedParametersBlock[i].second].push_back(firstParamNames[j]);
                }

            }
            
        }
        // remove the names of the first parameter from the list of names
        for (size_t i = 0; i < numOfSubParams; i++){
            paramsNames.erase(std::remove(paramsNames.begin(), paramsNames.end(), firstParamNames[i]), paramsNames.end());

        }
        
        it ++;
        
     }
    // add the independent parameters
    for (int i = 0; i < ChromosomeSubstitutionModel::NUM_OF_CHR_PARAMS; i++){
        uint numOfSubParams = LikelihoodUtils::getNumberOfParametersPerParamType(i, ChromEvolOptions::rateChangeType_);
        if (numOfSubParams == 0){ // parameter is ignored
            continue;
        }
        string paramBaseName = LikelihoodUtils::getStringParamName(i);
        for (uint j = 1; j <= numOfModels; j++){
            for (size_t k = 0; k < numOfSubParams; k++){
                string paramName;
                if (i == ChromosomeSubstitutionModel::BASENUM){
                    paramName = "Chromosome."+  paramBaseName + "_"+ std::to_string(j);
                }else{
                    paramName = "Chromosome."+  paramBaseName + std::to_string(k)+ "_"+ std::to_string(j);
                }
                if (std::find(paramsNames.begin(), paramsNames.end(), paramName) !=  paramsNames.end()){
                    mapOfTypeAndName[j][i].push_back(paramName);
                    pair<uint, int> paramModelAndType;
                    paramModelAndType.first = j;
                    paramModelAndType.second = i;
                    mapOfAliasedTypeModelAndParam[paramModelAndType].push_back(paramName);
                }
            }

        }
        
    }

    // print the values:
    for (uint m = 1; m <= numOfModels; m++){
        auto types = mapOfTypeAndName[m];
        for (int i = 0; i < ChromosomeSubstitutionModel::NUM_OF_CHR_PARAMS; i++){
            auto foundType = types.find(i);
            if (foundType == types.end()){
                continue;
            }
            for (size_t k = 0; k < mapOfTypeAndName[m][i].size(); k++){
                std::pair<uint, int> paramModelAndType;
                paramModelAndType.first = m;
                paramModelAndType.second = i;
                if (i != ChromosomeSubstitutionModel::BASENUM){
                
                    outFile << mapOfAliasedTypeModelAndParam[paramModelAndType][k] << " = "<< lik->getLikelihoodCalculation()->getParameter(mapOfTypeAndName[m][i][k]).getValue() << endl;

                }else{
                    outFile << mapOfAliasedTypeModelAndParam[paramModelAndType][k] << " = "<< (int)(lik->getLikelihoodCalculation()->getParameter(mapOfTypeAndName[m][i][k]).getValue()) << endl;
                }
            }        
        }
    }
}
/******************************************************************************/
double ChromosomeNumberMng::getOriginalTreeLength(string &path) const{
    Newick reader;
    auto originalTree = reader.readPhyloTree(path);
    double treeLength = originalTree->getTotalLength();
    delete originalTree;
    return treeLength;
}
