//
// File: ChromosomeNumberMng.h
// Created by: Anat Shafir
// Created on: Mon September 11 14:57 2020
//

/*
  Copyright or © or Copr. Bio++ Development Team, (November 16, 2004, 2005, 2006)

  This software is a computer program whose purpose is to provide classes
  for phylogenetic data analysis.

  This software is governed by the CeCILL  license under French law and
  abiding by the rules of distribution of free software.  You can  use,
  modify and/ or redistribute the software under the terms of the CeCILL
  license as circulated by CEA, CNRS and INRIA at the following URL
  "http://www.cecill.info".

  As a counterpart to the access to the source code and  rights to copy,
  modify and redistribute granted by the license, users are provided only
  with a limited warranty  and the software's author,  the holder of the
  economic rights,  and the successive licensors  have only  limited
  liability.

  In this respect, the user's attention is drawn to the risks associated
  with loading,  using,  modifying and/or developing or reproducing the
  software by the user in light of its specific status of free software,
  that may mean  that it is complicated to manipulate,  and  that  also
  therefore means  that it is reserved for developers  and  experienced
  professionals having in-depth computer knowledge. Users are therefore
  encouraged to load and test the software's suitability as regards their
  requirements in conditions enabling the security of their systems and/or
  data to be ensured and,  more generally, to use and operate it in the
  same conditions as regards security.

  The fact that you are presently reading this means that you have had
  knowledge of the CeCILL license and that you accept its terms.
*/
#ifndef CHROMEVOL_LIKELIHOODUTILS_H
#define CHROMEVOL_LIKELIHOODUTILS_H
#include <Bpp/Io/FileTools.h>
#include <Bpp/Text/TextTools.h>

#include <Bpp/Numeric/Prob/GammaDiscreteDistribution.h>
#include <Bpp/Phyl/Model/RateDistribution/GammaDiscreteRateDistribution.h>



//from bpp-seq

#include <Bpp/Seq/Alphabet/ChromosomeAlphabet.h>

//from bpp-phyl
#include <Bpp/Phyl/Tree/PhyloTreeTools.h>
#include <Bpp/Phyl/Likelihood/ParametrizablePhyloTree.h>
#include <Bpp/Phyl/Io/IoTree.h>
#include <Bpp/Phyl/Io/Newick.h>
#include <Bpp/Phyl/Likelihood/ParametrizablePhyloTree.h>
#include <Bpp/Phyl/Likelihood/NonHomogeneousSubstitutionProcess.h>
#include <Bpp/Phyl/Likelihood/PhyloLikelihoods/SingleProcessPhyloLikelihood.h>
#include "ChromosomeSubstitutionModel.h"
#include "TreeUtils.h"
#include "ChromEvolOptions.h"


//standard libraries
#include <string>
#include <vector>
#include <iostream>
#include <sys/stat.h>
#include <regex>


using namespace std;
namespace bpp{
  class LikelihoodUtils{
    public:
      LikelihoodUtils() {}
      virtual ~LikelihoodUtils() {}

    public:
        static void setNodeIdsForAllModels(PhyloTree* phyltree, std::map<uint, std::vector<uint>> &mapModelNodesIds, string &path, std::vector<uint> &initialModelNodes);
        static void getNodeIdsPerModelFromLine(string &content, PhyloTree* tree, std::map<uint, std::pair<uint, std::vector<uint>>> &modelAndNodeIds, std::map<uint,uint> &mapOriginalToAssignedModel, std::vector<uint> &initialModelNodes);
        static std::map<uint, std::vector<uint>> findMRCAForEachModelNodes(PhyloTree* tree, std::map<uint, vector<uint>> mapOfModelsAndNodes);
        static std::string getFunctionName(int func);
        static std::string getStringParamName(int type);
        static vector <double> setFixedRootFrequencies(const std::string &path, std::shared_ptr<ChromosomeSubstitutionModel> chrModel);
        static void updateWithTypeAndCorrespondingName(std::map<std::string, int> &typeGeneralName);
        static int getTypeOfParamFromParamName(string name);
        static uint getModelFromParamName(string name);
        static size_t getNumberOfFixedParams(SingleProcessPhyloLikelihood* lik, std::map<uint, vector<int>> &fixedParams);
        static uint getNumberOfParametersPerParamType(int paramType, vector<int> &funcTypes);
        static void updateMapsOfParamTypesAndNames(std::map<int, std::map<uint, std::vector<string>>> &typeWithParamNames, std::map<string, std::pair<int, uint>>* paramNameAndType, SingleProcessPhyloLikelihood* tl, std::map<int, std::vector<std::pair<uint, int>>>* sharedParams = 0);
        static void createMapOfSharedParameterNames(std::map<int, std::vector<std::pair<uint, int>>> &sharedParams, std::map<string, vector<std::pair<uint, int>>> &sharedParamsNames);
        static void getMutableMapOfModelAndNodeIds(std::map<uint, vector<uint>> &mapModelNodesIds, SingleProcessPhyloLikelihood* lik, uint rootId = 0);
        static std::map<uint, pair<int, std::map<int, std::vector<double>>>> getMapOfParamsForComplexModel(SingleProcessPhyloLikelihood* lik, std::map<int, std::map<uint, std::vector<string>>> typeWithParamNames, uint numOfModels);

        static void setParamsNameInForMultiProcess(std::map<uint, std::map<int, vector<string>>> &mapOfParamsNamesPerModelType, std::map<uint, pair<int, std::map<int, std::vector<double>>>> &modelParams);
        static void aliasParametersInSubstitutionProcess(std::map<uint, std::map<int, vector<string>>> &mapOfParamsNamesPerModelType, std::map<int, vector<std::pair<uint, int>>>* updatedSharedParams, std::shared_ptr<NonHomogeneousSubstitutionProcess> process);
        static bool compareLikValues(SingleProcessPhyloLikelihood* lik1, SingleProcessPhyloLikelihood* lik2);

        static SingleProcessPhyloLikelihood* setHeterogeneousModel(const PhyloTree* tree, const VectorSiteContainer* vsc, const ChromosomeAlphabet* alphabet, std::map<uint, uint> baseNumberUpperBound, std::map<uint, vector<uint>> &mapModelNodesIds, std::map<uint, pair<int, std::map<int, std::vector<double>>>> &modelParams, uint numOfModels, std::map<int, vector<std::pair<uint,int>>>* updatedSharedParams);
        static SingleProcessPhyloLikelihood* setRandomHeterogeneousModel(const PhyloTree* tree, const VectorSiteContainer* vsc, const ChromosomeAlphabet* alphabet, std::map<uint, uint> baseNumberUpperBound, std::map<uint, vector<uint>> &mapModelNodesIds, std::map<uint, pair<int, std::map<int, std::vector<double>>>> &modelParams, uint numOfModels, double parsimonyBound, std::map<uint, vector<int>> &fixedParams, std::map<int, std::vector<std::pair<uint, int>>>* sharedParams);



  };

}
#endif // CHROMEVOL_LIKELIHOODUTILS_H