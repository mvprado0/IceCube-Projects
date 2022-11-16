#include <stdio.h>
#include <iostream>
#include "TH1.h"
#include "TFile.h"
#include "TMinuit.h"
#include <errno.h>
#include <limits.h>
#include <TRandom3.h>
#include "TMath.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TTree.h"
#include "TLegend.h"
#include <complex.h>
#include "TH2.h"
#include "TGraph.h"
#include <vector>
#include <algorithm>
#include "TAxis.h"
#include <fstream>
#include "TMatrixT.h"
#include "TStyle.h"

void Rescaling(TH1D* hist){
    
    //Divides by the bin width
    for (int i=1; i<=hist->GetXaxis()->GetNbins(); i++) {
        double content = hist->GetBinContent(i);
        double binwidth = hist->GetBinWidth(i);
        hist->SetBinContent(i,content/binwidth);
    }
}

void Descaling(TH1D* hist){
    
    // Reverts normalization of the bin counts by multiplying back by the bin width.
    for (int i=1; i<=hist->GetXaxis()->GetNbins(); i++) {
	double content = hist->GetBinContent(i);
        double binwidth = hist->GetBinWidth(i);
        hist->SetBinContent(i,content*binwidth);
    }
}

void ProbMuE(TH1D* hist, double DeltaM2, double L, double theta_mue){
    
    //Muon neutrino to electron neutrino oscillation probability function multiplies histograms
    for (int i=1; i<=hist->GetXaxis()->GetNbins(); i++) {
        double EventOrig = hist->GetBinContent(i);
        double NuEnergy = hist->GetBinCenter(i);
        double Prob = theta_mue*pow(sin(DeltaM2*L/(4*3*pow(10,8)*NuEnergy*pow(10,9)*6.582119514*pow(10,-16))),2);
        hist->SetBinContent(i,EventOrig*Prob);
    }
}

void ProbEE(TH1D* hist, double DeltaM2, double L, double theta_ee){
    
    // Electron neutrino to electron neutrino oscillation probability function multiplies histograms
    for (int i=1; i<=hist->GetXaxis()->GetNbins(); i++) {
        double EventOrig = hist->GetBinContent(i);
        double NuEnergy = hist->GetBinCenter(i);
        double Prob = 1-theta_ee*pow(sin(DeltaM2*L/(4*3*pow(10,8)*NuEnergy*pow(10,9)*6.582119514*pow(10,-16))),2);
        
        //Subtracts survival nu_e's from total nu_e's to obtain how many oscillated
        double osc = EventOrig-EventOrig*Prob;
        hist->SetBinContent(i,osc);
    }
}

void ExpSignal(TH1D* hist1, TH1D* hist2, TH1D* signalhist,TH1D* back1,TH1D* back2,TH1D* dirt){
    
    //Adds events together for each bin to make the nu_e signal
    for (int i=1; i<=hist1->GetXaxis()->GetNbins(); i++) {
        double event1 = hist1->GetBinContent(i);
        double event2 = hist2->GetBinContent(i);
        double event3 = back1->GetBinContent(i);
        double event4 = back2->GetBinContent(i);
        double event5 = dirt->GetBinContent(i);
       
        double newcontent = event1+event2+event3+event4+event5;
        signalhist->SetBinContent(i,newcontent);
    }
}

double ChiSquare(TH1D* hist1, TH1D* hist2, TH1D* hist3, TH1D* hist4, TH1D* hist5, TH1D* hist6, TH1D* hist7, TH1D* hist8, TH1D* hist9, TH1D* sig1, TH1D* sig2, TH1D* sig3, TMatrixT<float>* xsec, TMatrixT<float>* flux, TH1D* back1SBND, TH1D* back2SBND, TH1D* back3MIC, TH1D* back4MIC, TH1D* back5IC, TH1D* back6IC, TH1D* dirtSBND, TH1D* dirtMIC, TH1D* dirtIC){
    
    double NumHist = hist1->GetXaxis()->GetNbins();
    
    std::vector <double> SBND;
    std::vector <double> MicroBooNE;
    std::vector <double> ICARUS;
    std::vector <double> DiffVec;
    std::vector <double> MCvar1;
    std::vector <double> MCvar2;
    std::vector <double> MCvar3;
    std::vector <double> MCAll;
    
    TMatrixT <float> Diff(NumHist*3,1);
    TMatrixT <float> Cov(NumHist*3,NumHist*3);
    TMatrixT <float> Dirt(NumHist*3,1);

    for (int i=1; i<=NumHist; i++) {
        
        double MC1 = hist1->GetBinContent(i);
        double MC2 = hist2->GetBinContent(i);
        double MC3 = hist3->GetBinContent(i);
        double MC4 = hist4->GetBinContent(i);
        double MC5 = hist5->GetBinContent(i);
        double MC6 = hist6->GetBinContent(i);
        double MC7 = hist7->GetBinContent(i);
        double MC8 = hist8->GetBinContent(i);
        double MC9 = hist9->GetBinContent(i);
        double MCb1 = back1SBND->GetBinContent(i);
        double MCb2 = back2SBND->GetBinContent(i);
        double MCb3 = back3MIC->GetBinContent(i);
        double MCb4 = back4MIC->GetBinContent(i);
        double MCb5 = back5IC->GetBinContent(i);
        double MCb6 = back6IC->GetBinContent(i);
        double MCd1 = dirtSBND->GetBinContent(i);
        double MCd2 = dirtMIC->GetBinContent(i);
        double MCd3 = dirtIC->GetBinContent(i);

        double MCSBND = MC1+MC2+MC3+MCb1+MCb2+MCd1;
        double MCMic = MC4+MC5+MC6+MCb3+MCb4+MCd2;
        double MCICARUS = MC7+MC8+MC9+MCb5+MCb6+MCd3;
        
        double DataSBND = sig1->GetBinContent(i);
        double DataMic = sig2->GetBinContent(i);
        double DataICARUS = sig3->GetBinContent(i);
        
        SBND.push_back(DataSBND-MCSBND);
        MicroBooNE.push_back(DataMic-MCMic);
        ICARUS.push_back(DataICARUS-MCICARUS);
        
        MCvar1.push_back(MCSBND);
        MCvar2.push_back(MCMic);
        MCvar3.push_back(MCICARUS);
    }
    
    DiffVec.insert(DiffVec.end(),SBND.begin(),SBND.end());
    DiffVec.insert(DiffVec.end(),MicroBooNE.begin(),MicroBooNE.end());
    DiffVec.insert(DiffVec.end(),ICARUS.begin(),ICARUS.end());
    
    MCAll.insert(MCAll.end(),MCvar1.begin(),MCvar1.end());
    MCAll.insert(MCAll.end(),MCvar2.begin(),MCvar2.end());
    MCAll.insert(MCAll.end(),MCvar3.begin(),MCvar3.end());
    
    for (int i=0; i<(NumHist*3); i++) {
        if (i==0||i==12||i==24) {
            Diff(i,0) = 0;
        } else {
            Diff(i,0) = DiffVec[i];
        }
    }
    
    for (int i=0; i<(NumHist*3); i++) {
        for (int j=0; j<(NumHist*3); j++) {
            if(i==j){
                Cov(j,i) = MCAll[i];
            } else {
                Cov(j,i)=0;
            }
        }
    }
    
    //Dirt uncertainty
    int g=0;
    
    for (int i=1; i<=NumHist; i++) {
        Dirt(g,0) = 0.15*dirtSBND->GetBinContent(i);
        g+=1;
    }
    
    for (int i=1; i<=NumHist; i++) {
        Dirt(g,0) = 0.15*dirtMIC->GetBinContent(i);
        g+=1;
    }
    
    for (int i=1; i<=NumHist; i++) {
        Dirt(g,0) = 0.15*dirtIC->GetBinContent(i);
        g+=1;
    }
    
    TMatrixT <float> DirtN = Dirt;
    TMatrixT <float> DirtCov = DirtN*Dirt.T();
    
    for (int i=0; i<(NumHist*3); i++) {
        for (int j=0; j<(NumHist*3); j++) {
            if (i>=(NumHist) && j<(NumHist)) {
                DirtCov(i,j)=0;
            } else if (i<(NumHist) && j>=(NumHist)) {
                DirtCov(i,j)=0;
            } else if (i>=(NumHist*2) && j<(NumHist*2)) {
                DirtCov(i,j)=0;
            } else if (i<(NumHist*2) && j>=(NumHist*2)) {
                DirtCov(i,j)=0;
            }
        }
    }
    
    //Back to calculating the total covariance matrix
    TMatrixT <float> Dorig = Diff;

    Cov += *flux + *xsec + DirtCov;

    TMatrixT <float> CovI = Cov.Invert();
    TMatrixT <float> DiffTrans = Diff.T();
    TMatrixT <float> chisq = DiffTrans*CovI*Dorig;
    
    return chisq(0,0);
    
}

void lsnd_plot (TCanvas* c, double* x1, double* x2, double* x3,double* x4, double* y1, double* y2, double* y3, double* y4, TGraph* gr[3]){
    c->cd();
    const char* data_dir = "lsnd_data/";
    Double_t  dm2BF[] = {1.2};
    Double_t sin22thBF[] = {0.003};
    
    const Int_t NDATAFILES = 11;
    const char * file_list[NDATAFILES] = {"llreg_608_1.vec",
     				          "llreg_608_2.vec",
    				          "llreg_608_3.vec",
    				          "llreg_607_1.vec",
    				          "llreg_607_2.vec",
    				          "llreg_607_3.vec",
    				          "llreg_607_4.vec",
    				          "llreg_607_5.vec",
    				          "llreg_607_6.vec",
    				          "llreg_607_7.vec",
    				          "llreg_607_8.vec"};
    
    Int_t graph_color[NDATAFILES] = {29, 29, 29, 38, 38, 38, 38, 38, 38, 38, 38};
    Int_t nlines;

    //Files have 500 inputs
    Double_t x[500],y[500];
    Double_t dummy, dummy_old;
    
    for (Int_t ifile = 0; ifile<NDATAFILES; ifile++) {
        nlines = 0;
        for (Int_t i=0;i<500;i++){
            x[i]=0.0;
            y[i]=0.0;
        }
        char  filename[100];
        strcpy(filename, data_dir);
        strcat(filename, file_list[ifile]);

        ifstream datafile;
        datafile.open(filename, ios_base::in);
        
        // Check if the file is open:
        if (!datafile.is_open() ) {
            std::cerr << "lsnd_plot.C: file not opened" <<std::endl; 
            return;
        }
        
        while (!datafile.eof()) {
            datafile >> dummy;
            datafile >> dummy;
            
            // Sine^2 values:
            datafile >> x[nlines];
            
            // Delta m^2 values:
            datafile >> y[nlines];
            
            nlines++;
            if (dummy == dummy_old) {
                nlines--;
            } 
            dummy_old = dummy;
        }
        
        if (ifile == 0) {
            std::memcpy(x1, x, 500 * sizeof(double));
            std::memcpy(y1, y, 500 * sizeof(double));
        } else if (ifile == 1) {
            std::memcpy(x2, x, 500 * sizeof(double));
            std::memcpy(y2, y, 500 * sizeof(double));
        } else if(ifile == 2){
            std::memcpy(x3, x, 500 * sizeof(double));
            std::memcpy(y3, y, 500 * sizeof(double));
        } else if(ifile == 3){
            std::memcpy(x4, x, 500 * sizeof(double));
            std::memcpy(y4, y, 500 * sizeof(double));
        }
        
        gr[ifile] = new TGraph(nlines,x,y);
        
        datafile.close();
    }
    
    std::cout << "Finished reading data files" << std::endl;
    
    for (Int_t ifile = 0; ifile<NDATAFILES; ifile++) {
        gr[ifile]->SetFillColor(graph_color[ifile]);
        gr[ifile]->Draw("LF");
    }

    //Add the best fit point;
    TGraph * bfPoint = new TGraph(1, sin22thBF, dm2BF);
    bfPoint -> SetLineColor(2);
    bfPoint -> SetMarkerStyle(3);
    bfPoint -> SetMarkerColor(1);
    bfPoint -> Draw("LP");
    
    TLegend* lg = new TLegend(0.59, 0.58, 0.88, 0.89);
    lg->AddEntry(gr[1], "LSND 99% CL");
    lg->AddEntry(gr[4], "LSND 90% CL");
    lg->AddEntry(bfPoint, "LSND Best Fit","p");
    
    lg->Draw();
   
    return;
}

void PrGLO(TCanvas* k, TGraph* gr[1]){

    k->cd();
    int nlines=0;
    double x[500],y[500];
    const char* data_dir = "lsnd_data/";
    const char * file_list[1] = {"prglo.vec"};
    
    for (int i=0;i<500;i++){
        x[i]=0.0;
        y[i]=0.0;
    }
    char  filename[100];
    strcpy(filename, data_dir);
    strcat(filename, file_list[0]);
    
    ifstream datafile;
    datafile.open(filename, ios_base::in);

    // Check if the file is open:
    if (!datafile.is_open() ) {std::cerr << "file not opened" <<std::endl; return;}
    while (!datafile.eof()) {
        datafile >> x[nlines];
        datafile >> y[nlines];
        nlines++;
    }
    
    datafile.close();
    
    gr[0] = new TGraph(nlines,x,y);
    gr[0]->SetFillColor(46);
    gr[0]->Draw("LF");
    
}

void signalhist(TH1D* hist1,TH1D* hist2,TH1D* hist3,TH1D* hist4,TH1D* hist5,TH1D* hist6,TH1D* hist7,TH1D* hist8,TH1D* hist9,TH1D* signal1,TH1D* signal2,TH1D* signal3, double m2, double L1, double L2, double L3, double theta_ee, double theta_mue, int NumBin, TH1D* expsignal1,TH1D* expsignal2,TH1D* expsignal3, TH1D* back1SBND, TH1D* back2SBND, TH1D* back3MIC, TH1D* back4MIC, TH1D* back5IC, TH1D* back6IC, TH1D* dirtSBND, TH1D* dirtMIC, TH1D* dirtIC, TH1D* nue_true1, TH1D* nue_true2, TH1D* nue_true3){
    
    double xbin[13]={ 0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.5, 1.75, 2.0, 3.0};

    //creates electron neutrino oscillated signal distributions for the three detectors that are not divided by the bin width for chi squared calculations
    TH1D* hnew1 = (TH1D*) nue_true1->Clone();
    ProbEE(hnew1,m2,L1,theta_ee);
    TH1D* hnew1R = (TH1D*) hnew1->Rebin(NumBin,"hnew1",xbin);

    for (int i=1; i<=hnew1R->GetXaxis()->GetNbins(); i++) {
        
        double cont1 = hist1->GetBinContent(i);
        double cont2 = hist2->GetBinContent(i);
        double cont3 = hist3->GetBinContent(i);
        double diff1 = hnew1R->GetBinContent(i);
        
        double newcont = cont1+cont2+cont3-diff1;
        
        hnew1R->SetBinContent(i,newcont);
    }
    
    TH1D* hnew2 = (TH1D*) nue_true2->Clone();
    ProbEE(hnew2,m2,L2,theta_ee);
    TH1D* hnew2R = (TH1D*) hnew2->Rebin(NumBin,"hnew1",xbin);
    
    for (int i=1; i<=hnew2R->GetXaxis()->GetNbins(); i++) {
        
        double cont1 = hist4->GetBinContent(i);
        double cont2 = hist5->GetBinContent(i);
        double cont3 = hist6->GetBinContent(i);
        double diff1 = hnew2R->GetBinContent(i);
        
        double newcont = cont1+cont2+cont3-diff1;
        
        hnew2R->SetBinContent(i,newcont);
    }
    
    TH1D* hnew3 = (TH1D*) nue_true3->Clone();
    ProbEE(hnew3,m2,L3,theta_ee);
    TH1D* hnew3R = (TH1D*) hnew3->Rebin(NumBin,"hnew1",xbin);
    
    for (int i=1; i<=hnew3R->GetXaxis()->GetNbins(); i++) {
        
        double cont1 = hist7->GetBinContent(i);
        double cont2 = hist8->GetBinContent(i);
        double cont3 = hist9->GetBinContent(i);
        double diff1 = hnew3R->GetBinContent(i);
        
        double newcont = cont1+cont2+cont3-diff1;
        
        hnew3R->SetBinContent(i,newcont);
    }
    
    TH1D* newsig1 = (TH1D*) signal1->Clone();
    ProbMuE(newsig1,m2,L1,theta_mue);
    TH1D* newsig1R = (TH1D*) newsig1->Rebin(NumBin,"signal1",xbin);
        
    TH1D* newsig2 = (TH1D*) signal2->Clone();
    ProbMuE(newsig2,m2,L2,theta_mue);
    TH1D* newsig2R = (TH1D*) newsig2->Rebin(NumBin,"signal2",xbin);
        
    TH1D* newsig3 = (TH1D*) signal3->Clone();
    ProbMuE(newsig3,m2,L3,theta_mue);
    TH1D* newsig3R = (TH1D*) newsig3->Rebin(NumBin,"signal3",xbin);
    
    ExpSignal(hnew1R,newsig1R,expsignal1,back1SBND,back2SBND,dirtSBND);
    ExpSignal(hnew2R,newsig2R,expsignal2,back3MIC,back4MIC,dirtMIC);
    ExpSignal(hnew3R,newsig3R,expsignal3,back5IC,back6IC,dirtIC);
    
}

void sigDraw(TH1D* expsignal1, TH1D* expsignal2, TH1D* expsignal3, double m2, double theta_mue, double theta_ee,TH1D* hist1,TH1D* hist2,TH1D* hist3,TH1D* hist4,TH1D* hist5,TH1D* hist6,TH1D* hist7,TH1D* hist8,TH1D* hist9,TH1D* signal1,TH1D* signal2,TH1D* signal3,double L1, double L2, double L3,int NumBin,THStack* hs,THStack* hs1,THStack* hs2,TH1D* back1SBND, TH1D* back2SBND, TH1D* back3MIC, TH1D* back4MIC, TH1D* back5IC, TH1D* back6IC, TH1D* dirtSBND, TH1D* dirtMIC, TH1D* dirtIC,TH1D* nue_true1, TH1D* nue_true2, TH1D* nue_true3,TLegend* legendPostition,TLegend* legendPostition2,TLegend* legendPostition3){
    
    TH1D* expClone1 = (TH1D*) expsignal1->Clone();
    TH1D* expClone2 = (TH1D*) expsignal2->Clone();
    TH1D* expClone3 = (TH1D*) expsignal3->Clone();
    
    TH1D* expCloneNo1 = (TH1D*) expsignal1->Clone();
    TH1D* expCloneNo2 = (TH1D*) expsignal2->Clone();
    TH1D* expCloneNo3 = (TH1D*) expsignal3->Clone();
   
    //Signal with electron neutrino disappearance
    signalhist(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, signal1, signal2, signal3, m2, L1, L2, L3, theta_ee, theta_mue, NumBin, expClone1, expClone2, expClone3,back1SBND, back2SBND, back3MIC, back4MIC, back5IC, back6IC, dirtSBND, dirtMIC, dirtIC, nue_true1, nue_true2, nue_true3);
    
    //Signal with no electron neutrino disappearance
    signalhist(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, signal1, signal2, signal3, m2, L1, L2, L3, 0, theta_mue, NumBin, expCloneNo1, expCloneNo2, expCloneNo3,back1SBND, back2SBND, back3MIC, back4MIC, back5IC, back6IC, dirtSBND, dirtMIC, dirtIC, nue_true1, nue_true2, nue_true3);
    
    //Divides electron neutrino oscillated signal by the bin width for plotting the distributions
    Rescaling(expClone1);
    Rescaling(expClone2);
    Rescaling(expClone3);
    
    Rescaling(expCloneNo1);
    Rescaling(expCloneNo2);
    Rescaling(expCloneNo3);

    TCanvas* canvasNUE = new TCanvas("cNuE","Electron Neutrino Distribution",200,20,600,2000);

    TPad* pad1 = new TPad("pad1","cSBND", .005, .005, .995, .995);
    pad1->Divide(1,3);

    pad1->Draw();
    pad1->cd(1);
    
    hs->Draw("H");
    expClone1->SetLineColor(kRed);
    expClone1->SetLineWidth(3);
    expClone1->Draw("SAME");
    expCloneNo1->SetLineColor(kBlack);
    expCloneNo1->SetLineWidth(2);
    expCloneNo1->SetLineStyle(7);
    expCloneNo1->Draw("SAME");
    
    hs->GetXaxis()->SetTitle("#nu_{e} Reconstructed Energy (GeV)");
    hs->GetXaxis()->SetTitleOffset(1);
    hs->GetXaxis()->SetTitleSize(.045);
    hs->GetYaxis()->SetTitle("Events per GeV");
    hs->GetYaxis()->SetTitleOffset(.8);
    hs->GetYaxis()->SetTitleSize(.045);
    hs->SetMaximum(21000);
    
    legendPostition->AddEntry(expClone1, "Oscillated Signal (#nu_{e} Disappearance)");
    legendPostition->AddEntry(expCloneNo1, "Oscillated Signal (No #nu_{e} Disappearance)");
    legendPostition->Draw();
    
    pad1->cd(2);
    
    hs1->Draw("H");
    expClone2->SetLineColor(kRed);
    expClone2->SetLineWidth(3);
    expClone2->Draw("SAME");
    expCloneNo2->SetLineColor(kBlack);
    expCloneNo2->SetLineWidth(2);
    expCloneNo2->SetLineStyle(7);
    expCloneNo2->Draw("SAME");
    
    hs1->GetXaxis()->SetTitle("#nu_{e} Reconstructed Energy (GeV)");
    hs1->GetXaxis()->SetTitleOffset(1);
    hs1->GetXaxis()->SetTitleSize(.045);
    hs1->GetYaxis()->SetTitle("Events per GeV");
    hs1->GetYaxis()->SetTitleOffset(.8);
    hs1->GetYaxis()->SetTitleSize(.045);    hs1->SetMaximum(1500);
    
    legendPostition2->AddEntry(expClone2, "Oscillated Signal (#nu_{e} Disappearance)");
    legendPostition2->AddEntry(expCloneNo2, "Oscillated Signal (No #nu_{e} Disappearance)");
    legendPostition2->Draw();
    
    pad1->cd(3);
    
    hs2->Draw("H");
    expClone3->SetLineColor(kRed);
    expClone3->SetLineWidth(3);
    expClone3->Draw("SAME");
    expCloneNo3->SetLineColor(kBlack);
    expCloneNo3->SetLineWidth(2);
    expCloneNo3->SetLineStyle(7);
    expCloneNo3->Draw("SAME");
    
    hs2->GetXaxis()->SetTitle("#nu_{e} Reconstructed Energy (GeV)");
    hs2->GetXaxis()->SetTitleOffset(1);
    hs2->GetXaxis()->SetTitleSize(.045);
    hs2->GetYaxis()->SetTitle("Events per GeV");
    hs2->GetYaxis()->SetTitleOffset(.8);
    hs2->GetYaxis()->SetTitleSize(.045);
    hs2->SetMaximum(3500);
    
    legendPostition3->AddEntry(expClone3, "Oscillated Signal (#nu_{e} Disappearance)");
    legendPostition3->AddEntry(expCloneNo3, "Oscillated Signal (No #nu_{e} Disappearance)");
    legendPostition3->Draw();
    
    canvasNUE->Modified();
    canvasNUE->Print("canvasNUE.pdf");
}

// Main

void NueDisappearance(){
    
    // Number of bins for the nu_e histograms.
    int NumBin = 12;
    
    // Detector distances in meters.
    double L1 = 100.0;
    double L2 = 470.0;
    double L3 = 600.0;

    // By theta_ee, it really means sin^2(2*theta_ee). If this value is zero, no electron neutrinos will oscillate and SBN sensitivity will not be reduced.
    double theta_ee = 0.1; 
    
    // Desired binning of electron neutrino event distributions.
    double xbin[13] = { 0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1, 1.3, 1.5, 1.75, 2.0, 3.0 };

    // Files needed.
    TFile* file00 = new TFile("complete_files/combined_ntuple_100m_nu_processed_numu.root","READ");
    TFile* file01 = new TFile("complete_files/combined_ntuple_470m_nu_processed_numu.root","READ");
    TFile* file02 = new TFile("complete_files/combined_ntuple_600m_onaxis_nu_processed_numu.root","READ");
    TFile* file03 = new TFile("complete_files/combined_ntuple_100m_nu_processed_nue.root","READ");
    TFile* file04 = new TFile("complete_files/combined_ntuple_470m_nu_processed_nue.root","READ");
    TFile* file05 = new TFile("complete_files/combined_ntuple_600m_onaxis_nu_processed_nue.root","READ");
    TFile* fluxsys = new TFile("complete_files/matrixFile_nue_ND_100m_uB_T600_onaxis_flux_6_ecalo2_nu_vePhot0.05_gap3.root","READ");
    TFile* xsecsys = new TFile("complete_files/matrixFile_nue_ND_100m_uB_T600_onaxis_xsec_0_ecalo2_nu_vePhot0.05_gap3.root","READ");

    // Electron neutrino events assuming no oscillations. Will need to be multiplied later by the oscillation probability to obtain neutrino events that oscillate to other flavors. 
    TH1D* nue_true1 = (TH1D*) file03->Get("NueTrueEnergy");
    TH1D* nue_true2 = (TH1D*) file04->Get("NueTrueEnergy");
    TH1D* nue_true3 = (TH1D*) file05->Get("NueTrueEnergy");

    // Muon neutrino events assuming no oscillations. Will need to be multiplied later by the oscillation probability to obtain the oscillated electron neutrino events.
    TH1D* signal1 = (TH1D*) file00->Get("NumuCC");
    TH1D* signal2 = (TH1D*) file01->Get("NumuCC");
    TH1D* signal3 = (TH1D*) file02->Get("NumuCC");

    // Neutrino flux and cross section systematics.
    TMatrixT <float>* flux = (TMatrixT <float>*) fluxsys->Get("covarianceMatrix");
    TMatrixT <float>* xsec = (TMatrixT <float>*) xsecsys->Get("covarianceMatrix");
    
    // Proposal files taken from Figure 21. These show the intrinsic electron neutrino events coming from the beam + background sources.  
    // Files order: mu, kplus, k0, NCsingleGamma, numuCC, dirt
    
    //SBND
    double file1[12] = { 0.0, 3859.3, 6010.1, 7276.4, 6251.3, 5025.1, 4040.2, 2894.5, 2030.2, 1346.7, 743.7, 201.0 };
    double file2[12] = { 0.0, 924.62, 2070.4, 2753.8, 3899.5, 4020.1, 4402.0, 4080.4, 3979.9, 3618.1, 2693.5, 1346.7 };
    double file3[12] = { 0.0, 361.81, 683.42, 984.92, 924.62, 1065.3, 1165.8, 1105.5, 783.92, 804.02, 603.01, 281.41 };
    double file4[12] = { 0.0, 3417.1, 2130.7, 783.92, 542.71, 281.41, 180.90, 100.50, 180.90, 140.70, 140.70, 40.201 };
    double file5[12] = { 0.0, 804.02, 683.42, 422.11, 180.90, 140.70, 20.101, 40.201, 40.201, 160.80, 120.60, 40.201 };
    double file6[12] = { 0.0, 120.60, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    //Microboone
    double file7[12] = { 0.0, 143.81, 248.88, 305.06, 270.70, 253.59, 219.22, 170.51, 131.88, 87.525, 53.240, 20.497 };
    double file8[12] = { 0.0, 61.801, 139.42, 169.61, 214.16, 238.61, 238.61, 222.80, 196.92, 166.73, 126.50, 67.500 };
    double file9[12] = { 0.0, 21.566, 35.937, 45.999, 56.063, 53.180, 48.868, 44.550, 43.123, 40.240, 31.609, 18.647 };
    double file10[12] = { 0.0, 196.92, 129.37, 60.375, 30.188, 18.686, 17.248, 11.497, 8.6257, 7.1869, 2.8703, 2.9249 };
    double file11[12] = { 0.0, 38.813, 40.240, 24.432, 22.996, 17.247, 14.372, 10.063, 2.8733, 8.6242, 10.069, 4.3062 };
    double file12[12] = { 0.0, 225.67, 64.682, 14.372, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    //ICARUS
    double file13[12] = { 0.0, 291.43, 456.80, 528.31, 505.97, 490.10, 383.62, 316.02, 209.63, 148.59, 90.814, 36.615 };
    double file14[12] = { 0.0, 90.608, 294.49, 375.40, 362.46, 385.11, 427.20, 385.11, 320.39, 281.55, 246.00, 116.26 };
    double file15[12] = { 0.0, 38.849, 74.434, 100.32, 97.080, 71.197, 64.704, 67.965, 84.153, 77.694, 64.742, 35.703 };
    double file16[12] = { 0.0, 375.40, 213.60, 103.57, 55.030, 48.551, 19.411, 19.407, 22.650, 29.130, 6.4760, 3.2118 };
    double file17[12] = { 0.0, 77.670, 71.204, 38.828, 32.349, 25.890, 22.654, 22.661, 0.0, 0.0, 0.0, 0.0 };
    double file18[12] = { 0.0, 310.67, 84.142, 22.654, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    //SBND histograms (Figure 21) - Signal and Background.
    // Signal
    TH1D* hist1 = new TH1D("sbnd1","NueFromMuonDecay_sbnd",NumBin,xbin);
    TH1D* hist2 = new TH1D("sbnd2","NueFromKPlusDecay_sbnd",NumBin,xbin);
    TH1D* hist3 = new TH1D("sbnd3","NueFromK0Decay_sbnd",NumBin,xbin);
    
    // Background
    TH1D* back1 = new TH1D("sbnd4","SinglePhotonNC_sbnd",NumBin,xbin);
    TH1D* back2 = new TH1D("sbnd5","NumuCC_sbnd",NumBin,xbin);
    TH1D* A1 = new TH1D("sbnd6","Dirt_sbnd",NumBin,xbin);

    //MicroBooNE
    TH1D* hist4 = new TH1D("mic1","NueFromMuonDecay_microboone",NumBin,xbin);
    TH1D* hist5 = new TH1D("mic2","NueFromKPlusDecay_microboone",NumBin,xbin);
    TH1D* hist6 = new TH1D("mic3","NueFromK0Decay_microboone",NumBin,xbin);
    TH1D* back3 = new TH1D("mic4","SinglePhotonNC_microboone",NumBin,xbin);
    TH1D* back4 = new TH1D("mic5","NumuCC_microboone",NumBin,xbin);
    TH1D* A2 = new TH1D("mic6","Dirt_microboone",NumBin,xbin);

    //ICARUS
    TH1D* hist7 = new TH1D("ic1","NueFromMuonDecay_icarus",NumBin,xbin);
    TH1D* hist8 = new TH1D("ic2","NueFromKPlusDecay_icarus",NumBin,xbin);
    TH1D* hist9 = new TH1D("ic3","NueFromK0Decay_icarus",NumBin,xbin);
    TH1D* back5 = new TH1D("ic4","SinglePhotonNC_icarus",NumBin,xbin);
    TH1D* back6 = new TH1D("ic5","NumuCC_icarus",NumBin,xbin);
    TH1D* A3 = new TH1D("ic6","Dirt_icarus",NumBin,xbin);

    for (int i=0; i<NumBin; i++) {
        hist1->SetBinContent(i+1,file1[i]);
        hist2->SetBinContent(i+1,file2[i]);
        hist3->SetBinContent(i+1,file3[i]);
        back1->SetBinContent(i+1,file4[i]);
        back2->SetBinContent(i+1,file5[i]);
        A1->SetBinContent(i+1,file6[i]);

        hist4->SetBinContent(i+1,file7[i]);
        hist5->SetBinContent(i+1,file8[i]);
        hist6->SetBinContent(i+1,file9[i]);
        back3->SetBinContent(i+1,file10[i]);
        back4->SetBinContent(i+1,file11[i]);
        A2->SetBinContent(i+1,file12[i]);

        hist7->SetBinContent(i+1,file13[i]);
        hist8->SetBinContent(i+1,file14[i]);
        hist9->SetBinContent(i+1,file15[i]);
        back5->SetBinContent(i+1,file16[i]);
        back6->SetBinContent(i+1,file17[i]);
        A3->SetBinContent(i+1,file18[i]);
    }
    
    // Normalization of bin content needs to be reverted for chi squared calculation.
    Descaling(hist1);
    Descaling(hist2);
    Descaling(hist3);
    Descaling(back1);
    Descaling(back2);
    Descaling(A1);

    Descaling(hist4);
    Descaling(hist5);
    Descaling(hist6);
    Descaling(back3);
    Descaling(back4);
    Descaling(A2);

    Descaling(hist7);
    Descaling(hist8);
    Descaling(hist9);
    Descaling(back5);
    Descaling(back6);
    Descaling(A3);
    
    double numberS = signal1->GetXaxis()->GetNbins();

    // Normalization factors for the electron neutrino unoscillated true energy distributions with respect to Figure 21:
    for (int i=1; i<=numberS; i++) {
        double oldcont = nue_true1->GetBinContent(i);
        double newcont = oldcont*0.321223;
        nue_true1->SetBinContent(i,newcont);
    }

    for (int i=1; i<=numberS; i++) {
        double oldcont = nue_true2->GetBinContent(i);
        double newcont = oldcont*0.614431;
        nue_true2->SetBinContent(i,newcont);
    }

    for (int i=1; i<=numberS; i++) {
        double oldcont = nue_true3->GetBinContent(i);
        double newcont = oldcont*0.301415;
        nue_true3->SetBinContent(i,newcont);
    }
    
    // Normalization factors for the muon neutrino unoscillated true energy distribution files with respect to Figure 24:
    for (int i=1; i<=numberS; i++) {
        double oldcont = signal1->GetBinContent(i);
        double newcont = oldcont*0.6712384025816862;
        signal1->SetBinContent(i,newcont);
    }

    for (int i=1; i<=numberS; i++) {
        double oldcont = signal2->GetBinContent(i);
        double newcont = oldcont*0.9991437154327747*2;
        signal2->SetBinContent(i,newcont);
    }

    for (int i=1; i<=numberS; i++) {
        double oldcont = signal3->GetBinContent(i);
        double newcont = oldcont*1.0119457119886637;
        signal3->SetBinContent(i,newcont);
    }
    
    // Cloning histograms, otherwise ROOT will replace the original histogram with any modified version.
    TH1D* hist1clone = (TH1D*) hist1->Clone();
    TH1D* hist2clone = (TH1D*) hist2->Clone();
    TH1D* hist3clone = (TH1D*) hist3->Clone();
    TH1D* hist4clone = (TH1D*) hist4->Clone();
    TH1D* hist5clone = (TH1D*) hist5->Clone();
    TH1D* hist6clone = (TH1D*) hist6->Clone();
    TH1D* hist7clone = (TH1D*) hist7->Clone();
    TH1D* hist8clone = (TH1D*) hist8->Clone();
    TH1D* hist9clone = (TH1D*) hist9->Clone();
    
    TH1D* clone1CHI = (TH1D*) hist1->Clone();
    TH1D* clone2CHI = (TH1D*) hist2->Clone();
    TH1D* clone3CHI = (TH1D*) hist3->Clone();
    TH1D* clone4CHI = (TH1D*) hist4->Clone();
    TH1D* clone5CHI = (TH1D*) hist5->Clone();
    TH1D* clone6CHI = (TH1D*) hist6->Clone();
    TH1D* clone7CHI = (TH1D*) hist7->Clone();
    TH1D* clone8CHI = (TH1D*) hist8->Clone();
    TH1D* clone9CHI = (TH1D*) hist9->Clone();

    TH1D* back1clone = (TH1D*) back1->Clone();
    TH1D* back2clone = (TH1D*) back2->Clone();
    TH1D* back3clone = (TH1D*) back3->Clone();
    TH1D* back4clone = (TH1D*) back4->Clone();
    TH1D* back5clone = (TH1D*) back5->Clone();
    TH1D* back6clone = (TH1D*) back6->Clone();
    
    TH1D* dirt1clone = (TH1D*) A1->Clone();
    TH1D* dirt2clone = (TH1D*) A2->Clone();
    TH1D* dirt3clone = (TH1D*) A3->Clone();
    
    TCanvas* canvas = new TCanvas("canvas","Histogram1",200,20,1000,600);
    
    //Divides by the bin width.
    //SBND nu_e intrinsics and backgrounds
    Rescaling(hist1clone);
    Rescaling(hist2clone);
    Rescaling(hist3clone);
    Rescaling(back1clone);
    Rescaling(back2clone);
    Rescaling(dirt1clone);
    hist1clone->SetFillColor(32);
    hist2clone->SetFillColor(30);
    hist3clone->SetFillColor(kGreen);
    back1clone->SetFillColor(42);
    back2clone->SetFillColor(38);
    dirt1clone->SetFillColor(17);
   
    // MicroBooNE nu_e intrinsics and backgrounds
    Rescaling(hist4clone);
    Rescaling(hist5clone);
    Rescaling(hist6clone);
    Rescaling(back3clone);
    Rescaling(back4clone);
    Rescaling(dirt2clone);
    hist4clone->SetFillColor(32);
    hist5clone->SetFillColor(30);
    hist6clone->SetFillColor(kGreen);
    back3clone->SetFillColor(42);
    back4clone->SetFillColor(38);
    dirt2clone->SetFillColor(17);
    
    // ICARUS nu_e intrinsics and backgrounds
    Rescaling(hist7clone);
    Rescaling(hist8clone);
    Rescaling(hist9clone);
    Rescaling(back5clone);
    Rescaling(back6clone);
    Rescaling(dirt3clone);
    hist7clone->SetFillColor(32);
    hist8clone->SetFillColor(30);
    hist9clone->SetFillColor(kGreen);
    back5clone->SetFillColor(42);
    back6clone->SetFillColor(38);
    dirt3clone->SetFillColor(17);
    
    // Stacks the rescaled histograms
    THStack *hs = new THStack("hs","SBND (100m)");
    
    hs->Add(hist1clone);
    hs->Add(hist2clone);
    hs->Add(hist3clone);
    hs->Add(back1clone);
    hs->Add(back2clone);
    hs->Add(dirt1clone);
    hs->Draw("H");
    
    hs->GetXaxis()->SetTitle("#nu_{e} Reconstructed Energy (GeV)");
    hs->GetXaxis()->SetTitleOffset(1);
    hs->GetYaxis()->SetTitle("Events per GeV");
    hs->GetYaxis()->SetTitleOffset(1.5);
    hs->SetMaximum(21000);
    
    TLegend* legendPostition = new TLegend(0.59, 0.58, 0.88, 0.89);
    legendPostition->AddEntry(hist1clone, "#mu #rightarrow #nu_{e}");
    legendPostition->AddEntry(hist2clone, "K^{+} #rightarrow #nu_{e}");
    legendPostition->AddEntry(hist3clone, "K^{0} #rightarrow #nu_{e}");
    legendPostition->AddEntry(back1clone, "NC single #gamma");
    legendPostition->AddEntry(back2clone, "#nu_{#mu} CC");
    legendPostition->AddEntry(dirt1clone, "Dirt");

    legendPostition->Draw();
    
    canvas->Modified();
    canvas->Print("canvas.pdf");

    TCanvas* canvas1 = new TCanvas("canvas1","Histogram2",200,20,1000,600);
    
    THStack *hs1 = new THStack("hs1","MicroBooNE (470m)");
    
    hs1->Add(hist4clone);
    hs1->Add(hist5clone);
    hs1->Add(hist6clone);
    hs1->Add(back3clone);
    hs1->Add(back4clone);
    hs1->Add(dirt2clone);
    hs1->Draw("H");
    
    hs1->GetXaxis()->SetTitle("#nu_{e} Reconstructed Energy (GeV)");
    hs1->GetXaxis()->SetTitleOffset(1);
    hs1->GetYaxis()->SetTitle("Events per GeV");
    hs1->GetYaxis()->SetTitleOffset(1.5);
    hs1->SetMaximum(1500);
    
    TLegend* legendPostition2 = new TLegend(0.59, 0.58, 0.88, 0.89);
    legendPostition2->AddEntry(hist4clone, "#mu #rightarrow #nu_{e}");
    legendPostition2->AddEntry(hist5clone, "K^{+} #rightarrow #nu_{e}");
    legendPostition2->AddEntry(hist6clone, "K^{0} #rightarrow #nu_{e}");
    legendPostition2->AddEntry(back3clone, "NC single #gamma");
    legendPostition2->AddEntry(back4clone, "#nu_{#mu} CC");
    legendPostition2->AddEntry(dirt2clone, "Dirt");
    legendPostition2->Draw();
    
    canvas1->Modified();
    canvas->Print("canvas1.pdf");
    
    TCanvas* canvas2 = new TCanvas("canvas2","Histogram3",200,20,1000,600);
    THStack *hs2 = new THStack("hs2","ICARUS-T600 (600m)");
    
    hs2->Add(hist7clone);
    hs2->Add(hist8clone);
    hs2->Add(hist9clone);
    hs2->Add(back5clone);
    hs2->Add(back6clone);
    hs2->Add(dirt3clone);
    hs2->Draw("H");
    
    hs2->GetXaxis()->SetTitle("#nu_{e} Reconstructed Energy (GeV)");
    hs2->GetXaxis()->SetTitleOffset(1);
    hs2->GetYaxis()->SetTitle("Events per GeV");
    hs2->GetYaxis()->SetTitleOffset(1.5);
    hs2->SetMaximum(3500);
    
    TLegend* legendPostition3 = new TLegend(0.59, 0.58, 0.88, 0.89);
    legendPostition3->AddEntry(hist7clone, "#mu #rightarrow #nu_{e}");
    legendPostition3->AddEntry(hist8clone, "K^{+} #rightarrow #nu_{e}");
    legendPostition3->AddEntry(hist9clone, "K^{0} #rightarrow #nu_{e}");
    legendPostition3->AddEntry(back5clone, "NC single #gamma");
    legendPostition3->AddEntry(back6clone, "#nu_{#mu} CC");
    legendPostition3->AddEntry(dirt3clone, "Dirt");
    legendPostition3->Draw();
    
    canvas2->Modified();
    canvas->Print("canvas2.pdf");

    //MARK: LSND
    //LSND PLOT
    int lsndsize = 500;
    
    double x1[lsndsize];
    double x2[lsndsize];
    double x3[lsndsize];
    double x4[lsndsize];
    double y1[lsndsize];
    double y2[lsndsize];
    double y3[lsndsize];
    double y4[lsndsize];
    int NDATAFILES = 11;
    
    TCanvas * d = new TCanvas("LSND Region", "LSND Region", 200,20,1000,600);
    d->SetLogx();
    d->SetLogy();
    
    TH2D* hr1=new TH2D("hr1","hr1",500,0.0001,1,500,0.01,100);
    hr1->Reset();
    hr1->SetFillColor(0);
    hr1->SetTitle(";sin^{2}2#theta_{#mue};#Deltam^{2} (eV^{2})");
    hr1->SetStats(kFALSE);
    hr1->Draw();
    
    TGraph* gr[NDATAFILES];
    
    lsnd_plot(d,x1,x2,x3,x4,y1,y2,y3,y4,gr);
    d->Print("lsnd.pdf");
    
    //MARK: PrGLO
    //PrGLO allowed region
    TCanvas * pr = new TCanvas("PrGLO Region", "PrGLO Region", 200,20,1000,600);
    pr->SetLogx();
    pr->SetLogy();
    
    TH2D* hpr=new TH2D("hpr","hpr",500,0.0001,1,500,0.01,100);
    hpr->Reset();
    hpr->SetFillColor(0);
    hpr->SetTitle(";sin^{2}2#theta_{#mue};#Deltam^{2} (eV^{2})");
    hpr->SetStats(kFALSE);
    hpr->Draw();
    
    TGraph* region[1];
    
    PrGLO(pr,region);
    pr->Print("prglo.pdf");
    
    TCanvas* canvas3 = new TCanvas("canvas3","Histogram4",200,20,435,415);
    
    //Complete electron neutrino signal histograms
    TH1D* expsignal1 = new TH1D("signal","Data Signal SBND (100m)",NumBin,xbin);
    TH1D* expsignal2 = new TH1D("signal2","Data Signal MicroBooNE (470m)",NumBin,xbin);
    TH1D* expsignal3 = new TH1D("signal3","Data Signal ICARUS-T600 (600m)",NumBin,xbin);
    
    //Min and max values for delta m^2 and sin(2*theta_mue)^2 in chi squared plot
    double M2min = pow(10,-2);
    double Thmin = pow(10,-4);
    double M2max = 100;
    double Thmax = 1;
    
    //Min and max values for delta m^2 and sin(2*theta_ee)^2 in chi squared plot
    double M2min3 = pow(10,-1);
    double Thmin3 = pow(10,-3);
    double M2max3 = 100;
    double Thmax3 = 1;
    
    //MARK: Number of Steps
    //Number of bins in the chi squared plot for the x and y axes
    int numberStepsx = 20;
    int numberStepsy = 20;
    
    std::vector <double> MSq;
    std::vector <double> MSq3;
    std::vector <double> Th;
    std::vector <double> Th3;
    std::vector <double> MSqC;
    std::vector <double> MSqC3;
    std::vector <double> ThC;
    std::vector <double> ThC3;
    std::vector <double> ChiCheck;
    std::vector <double> ChiCheck2;
    std::vector <double> ChiCheck3;
    
    for (int i=0; i<numberStepsy; i++) {
        for (int j=0; j<numberStepsx; j++) {

            //log binning to span the entire chi square surface
            double m2 = pow(10., (TMath::Log10(M2min)+ (i * (1./(numberStepsy-1)))*TMath::Log10(M2max/M2min)));
            double theta_mue = pow(10., (TMath::Log10(Thmin)+ (j * (1./(numberStepsx-1)))*TMath::Log10(Thmax/Thmin)));
            
            if (std::find(MSq.begin(), MSq.end(), m2) == MSq.end()){
                MSq.push_back(m2);
            }
            
            if (std::find(Th.begin(), Th.end(), theta_mue) == Th.end()){
                Th.push_back(theta_mue);
            }
            
            //Now log binning for delta m^2 vs sin(2*theta_ee)^2 chi squared plot
            double m23 = pow(10., (TMath::Log10(M2min3)+ (i * (1./(numberStepsy-1)))*TMath::Log10(M2max3/M2min3)));
            double theta_mue3 = pow(10., (TMath::Log10(Thmin3)+ (j * (1./(numberStepsx-1)))*TMath::Log10(Thmax3/Thmin3)));
            
            if (std::find(MSq3.begin(), MSq3.end(), m23) == MSq3.end()){
                MSq3.push_back(m23);
            }
            
            if (std::find(Th3.begin(), Th3.end(), theta_mue3) == Th3.end()){
                Th3.push_back(theta_mue3);
            }
        }
    }
    
    std::sort(Th.begin(), Th.end());
    std::sort(MSq.begin(), MSq.end());
    
    std::sort(Th3.begin(), Th3.end());
    std::sort(MSq3.begin(), MSq3.end());
    
    TH2D* ChiSqSurface = new TH2D("chi","Chi Square Surface",numberStepsx-1,&Th[0],numberStepsy-1,&MSq[0]);
    TH2D* ChiSqSurface2 = new TH2D("chi0","Sensitivity Plane (sin^{2}(2 #theta_{ee})=0.1)",numberStepsx-1,&Th[0],numberStepsy-1,&MSq[0]);
    TH2D* ChiSqSurfaceEE = new TH2D("chiEE","Sensitivity Plane (sin^{2}(2 #theta_{#mu e})=0.0013)",numberStepsx-1,&Th3[0],numberStepsy-1,&MSq3[0]);
    
    //Gets the center of each bin
    for (int j=1; j<numberStepsx; j++) {
        double centerT = ChiSqSurface->GetXaxis()->GetBinCenterLog(j);
        double centerT3 = ChiSqSurfaceEE->GetXaxis()->GetBinCenterLog(j);

        ThC.push_back(centerT);
        ThC3.push_back(centerT3);
    }
    
    for (int i=1; i<numberStepsy; i++) {
        double centerM = ChiSqSurface->GetYaxis()->GetBinCenterLog(i);
        double centerM3 = ChiSqSurfaceEE->GetYaxis()->GetBinCenterLog(i);

        MSqC.push_back(centerM);
        MSqC3.push_back(centerM3);
    }
    
    int MSqCsize = MSqC.size();
    int ThCsize = ThC.size();
  
    int MSqCsize3 = MSqC3.size();
    int ThCsize3 = ThC3.size();
    
    // Computes the chi squared
    for (int i=0; i<MSqCsize; i++) {
        for (int j=0; j<ThCsize; j++) {
            TH1D* expClone1 = (TH1D*) expsignal1->Clone();
            TH1D* expClone2 = (TH1D*) expsignal2->Clone();
            TH1D* expClone3 = (TH1D*) expsignal3->Clone();

            signalhist(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, signal1, signal2, signal3, MSqC[i], L1, L2, L3, theta_ee, ThC[j], NumBin, expClone1, expClone2, expClone3, back1, back2, back3, back4, back5, back6, A1, A2, A3, nue_true1, nue_true2, nue_true3);
            
            double chisq = ChiSquare(clone1CHI, clone2CHI, clone3CHI, clone4CHI, clone5CHI, clone6CHI, clone7CHI, clone8CHI, clone9CHI, expClone1, expClone2, expClone3, xsec, flux, back1, back2, back3, back4, back5, back6, A1, A2, A3);
            
            ChiCheck.push_back(chisq);
        }
    }
    
    int g = 0;
    
    for (int i=1; i<=MSqCsize; i++) {
        for (int j=1; j<=ThCsize; j++) {
            ChiSqSurface->SetBinContent(j,i,ChiCheck[g]);
            g = g+1;
        }
    }
    
    //Now the chi squared at sin(2*theta_ee)^2=0 to superimpose plots and show shift in contours.
    for (int i=0; i<MSqCsize; i++) {
        for (int j=0; j<ThCsize; j++) {
            TH1D* expClone1 = (TH1D*) expsignal1->Clone();
            TH1D* expClone2 = (TH1D*) expsignal2->Clone();
            TH1D* expClone3 = (TH1D*) expsignal3->Clone();
            
            signalhist(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, signal1, signal2, signal3, MSqC[i], L1, L2, L3, 0, ThC[j], NumBin, expClone1, expClone2, expClone3, back1, back2, back3, back4, back5, back6, A1, A2, A3, nue_true1, nue_true2, nue_true3);
            
            double chisq = ChiSquare(clone1CHI, clone2CHI, clone3CHI, clone4CHI, clone5CHI, clone6CHI, clone7CHI, clone8CHI, clone9CHI, expClone1, expClone2, expClone3, xsec, flux, back1, back2, back3, back4, back5, back6, A1, A2, A3);
            
            ChiCheck2.push_back(chisq);
        }
    }
    
    int k = 0;
    
    for (int i=1; i<=MSqCsize; i++) {
        for (int j=1; j<=ThCsize; j++) {
            ChiSqSurface2->SetBinContent(j,i,ChiCheck2[k]);
            k = k+1;
        }
    }
    
    //MARK: Contours
    double contours[3]={1.64,7.74,23.40};
    int colors[3] = {903, kAzure, 419};
    
    canvas3->SetLogx();
    canvas3->SetLogy();
    canvas3->SetLogz();
    
    ChiSqSurface2->GetXaxis()->SetTitle("sin^{2}(2 #theta_{#mu e})");
    ChiSqSurface2->GetYaxis()->SetTitle("#Delta m^{2} (eV^{2})");
    ChiSqSurface2->GetXaxis()->SetTitleOffset(1.4);
    ChiSqSurface2->GetYaxis()->SetTitleOffset(1.4);
    
    gStyle->SetPalette(3,colors);
    ChiSqSurface2->Smooth();
    ChiSqSurface2->SetContour(3,contours);
    ChiSqSurface2->Draw("cont1 same");
    ChiSqSurface2->SetLineWidth(3);
    ChiSqSurface2->SetLineStyle(7);
    
    for (int i = 0; i<11; i++) {
        int graph_color[11] = {29, 29, 29, 38, 38, 38, 38, 38, 38, 38, 38};
        gr[i]->SetFillColor(graph_color[i]);
        gr[i]->Draw("LF same");
    }
    
    region[0]->SetFillColor(397);
    region[0]->Draw("LF same");
    
    ChiSqSurface->Smooth();
    ChiSqSurface->SetContour(3,contours);
    ChiSqSurface->Draw("cont1 same");
    ChiSqSurface->SetLineWidth(4);
    
    //LSND best fit point
    double dm2BF[] = {1.2};
    double sin22thBF[] = {0.003};
    
    TGraph * bfPoint = new TGraph(1, sin22thBF, dm2BF);
    bfPoint -> SetLineColor(2);
    bfPoint -> SetMarkerStyle(3);
    bfPoint -> SetMarkerColor(1);
    bfPoint -> Draw("LP same");
    
    //PrGLO best fit point
    double PrGm[] = {1.6};
    double PrGs[] = {0.0013};
    
    TGraph * PrbfPoint = new TGraph(1, PrGs, PrGm);
    PrbfPoint -> SetLineColor(4);
    PrbfPoint -> SetMarkerStyle(3);
    PrbfPoint -> SetMarkerColor(634);
    PrbfPoint -> Draw("LP same");

    TLegend* lg1 = new TLegend(0.59, 0.58, 0.88, 0.89);
    lg1->AddEntry(gr[1], "LSND 99% CL");
    lg1->AddEntry(gr[4], "LSND 90% CL");
    lg1->AddEntry(bfPoint, "LSND Best Fit","p");
    lg1->AddEntry(region[0], "3+1 PrGLO 3#sigma CL");
    lg1->AddEntry(PrbfPoint, "3+1 PrGLO Best Fit","p");

    lg1->Draw();
    
    canvas3->Modified();
    
    canvas3->Print("ChiSq.pdf");
    
    // Computes the delta m^2 vs sin(2*theta_ee)^2 chi squared for a value of sin(2*theta_mue)^2=0.0013.
    double theta_mue3=0.0013;
    TCanvas* canvas5 = new TCanvas("canvas5","Histogram6",200,20,1000,600);

    for (int i=0; i<MSqCsize3; i++) {
        for (int j=0; j<ThCsize3; j++) {
            TH1D* expClone1 = (TH1D*) expsignal1->Clone();
            TH1D* expClone2 = (TH1D*) expsignal2->Clone();
            TH1D* expClone3 = (TH1D*) expsignal3->Clone();
            
            signalhist(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, signal1, signal2, signal3, MSqC3[i], L1, L2, L3, ThC3[j], theta_mue3, NumBin, expClone1, expClone2, expClone3, back1, back2, back3, back4, back5, back6, A1, A2, A3, nue_true1, nue_true2, nue_true3);
            
            double chisq = ChiSquare(clone1CHI, clone2CHI, clone3CHI, clone4CHI, clone5CHI, clone6CHI, clone7CHI, clone8CHI, clone9CHI, expClone1, expClone2, expClone3, xsec, flux, back1, back2, back3, back4, back5, back6, A1, A2, A3);
            
            ChiCheck3.push_back(chisq);
        }
    }
    
    int r = 0;
    
    for (int i=1; i<=MSqCsize3; i++) {
        for (int j=1; j<=ThCsize3; j++) {
            ChiSqSurfaceEE->SetBinContent(j,i,ChiCheck3[r]);
            r = r+1;
        }
    }
    
    canvas5->SetLogx();
    canvas5->SetLogy();
    canvas5->SetLogz();
    
    ChiSqSurfaceEE->GetXaxis()->SetTitle("sin^{2}(2 #theta_{ee})");
    ChiSqSurfaceEE->GetYaxis()->SetTitle("#Delta m^{2} (eV^{2})");
    
    ChiSqSurfaceEE->Smooth();
    ChiSqSurfaceEE->SetContour(3,contours);
    ChiSqSurfaceEE->Draw("cont1 same");
    ChiSqSurfaceEE->SetLineWidth(4);
    
    // Draws black box of preferred region
    double x_ee[5]={0.04,0.04,0.2,0.2,0.04};
    double y_ee[5]={0.8,2.0,2.0,0.8,0.8};
    
    TGraph* gr_ee = new TGraph(5,x_ee,y_ee);
    gr_ee->SetLineColor(kBlack);
    gr_ee->SetLineWidth(6);
    gr_ee->Draw("L same");
    
    canvas5->Modified();
    canvas5->Print("canvas5.pdf");
    
    int count1 = 0;
    int count2 = 0;
    int count3 = 0;
    int count4 = 0;    
    
    std::vector <double> lsndx;
    std::vector <double> lsndy;
    std::vector <double> lsndx90;
    std::vector <double> lsndy90;
    std::vector <double> lsndsig;
    std::vector <double> lsndsig0;
    std::vector <double> lsndsig90;
    std::vector <double> lsndsig900;
    
    //Gets the LSND 99% and 90% CL left edges
    for (int i=0; i<500; i++) {
        if (y1[i]<y1[i+1]) {
            count1++;
        }
        
        if(y2[i]<y2[i+1]) {
            count2++;
        }
        
        if(y3[i]<y3[i+1]) {
            count3++;
        }
        
        if(y4[i]<y4[i+1]) {
            count4++;
        }
    }
    
    //99% CL left edge
    for (int i=0; i<count1; i++) {
        lsndx.push_back(x1[i]);
        lsndy.push_back(y1[i]);
    }
    
    for (int i=0; i<count2; i++) {
        lsndx.push_back(x2[i]);
        lsndy.push_back(y2[i]);
    }
    
    for (int i=0; i<count3; i++) {
        lsndx.push_back(x3[i]);
        lsndy.push_back(y3[i]);
    }
    
    //90% CL left edge
    for (int i=0; i<count4; i++) {
        lsndx90.push_back(x4[i]);
        lsndy90.push_back(y4[i]);
    }
    
    int lsndxS = lsndx.size();
    int lsndyS = lsndy.size();
    int lsndyS90 = lsndy90.size();
    
    //LSND 99% CL left edge significance with theta_ee original value in beginning of code
    for (int i=0; i<lsndyS; i++) {
        TH1D* expClone1 = (TH1D*) expsignal1->Clone();
        TH1D* expClone2 = (TH1D*) expsignal2->Clone();
        TH1D* expClone3 = (TH1D*) expsignal3->Clone();
        
        signalhist(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, signal1, signal2, signal3, lsndy[i], L1, L2, L3, theta_ee, lsndx[i], NumBin, expClone1, expClone2, expClone3, back1, back2, back3, back4, back5, back6, A1, A2, A3, nue_true1, nue_true2, nue_true3);
        
        double chisq = ChiSquare(clone1CHI, clone2CHI, clone3CHI, clone4CHI, clone5CHI, clone6CHI, clone7CHI, clone8CHI, clone9CHI, expClone1, expClone2, expClone3,xsec,flux, back1, back2, back3, back4, back5, back6, A1, A2, A3);
       
        lsndsig.push_back(sqrt(chisq));
    }
    
    //LSND 99% CL left edge significance setting theta_ee=0
    for (int i=0; i<lsndyS; i++) {
        TH1D* expClone1 = (TH1D*) expsignal1->Clone();
        TH1D* expClone2 = (TH1D*) expsignal2->Clone();
        TH1D* expClone3 = (TH1D*) expsignal3->Clone();
        
        signalhist(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, signal1, signal2, signal3, lsndy[i], L1, L2, L3, 0, lsndx[i], NumBin, expClone1, expClone2, expClone3, back1, back2, back3, back4, back5, back6, A1, A2, A3, nue_true1, nue_true2, nue_true3);
        
        double chisq = ChiSquare(clone1CHI, clone2CHI, clone3CHI, clone4CHI, clone5CHI, clone6CHI, clone7CHI, clone8CHI, clone9CHI, expClone1, expClone2, expClone3,xsec,flux, back1, back2, back3, back4, back5, back6, A1, A2, A3);
        
        lsndsig0.push_back(sqrt(chisq));
    }
    
    //LSND 90% CL left edge significance with theta_ee original value in beginning of code
    for (int i=0; i<lsndyS90; i++) {
        TH1D* expClone1 = (TH1D*) expsignal1->Clone();
        TH1D* expClone2 = (TH1D*) expsignal2->Clone();
        TH1D* expClone3 = (TH1D*) expsignal3->Clone();
        
        signalhist(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, signal1, signal2, signal3, lsndy90[i], L1, L2, L3, theta_ee, lsndx90[i], NumBin, expClone1, expClone2, expClone3, back1, back2, back3, back4, back5, back6, A1, A2, A3, nue_true1, nue_true2, nue_true3);
        
        double chisq = ChiSquare(clone1CHI, clone2CHI, clone3CHI, clone4CHI, clone5CHI, clone6CHI, clone7CHI, clone8CHI, clone9CHI, expClone1, expClone2, expClone3,xsec,flux, back1, back2, back3, back4, back5, back6, A1, A2, A3);
        
        lsndsig90.push_back(sqrt(chisq));
    }

    //LSND 90% CL left edge significance setting theta_ee=0
    for (int i=0; i<lsndyS90; i++) {
        TH1D* expClone1 = (TH1D*) expsignal1->Clone();
        TH1D* expClone2 = (TH1D*) expsignal2->Clone();
        TH1D* expClone3 = (TH1D*) expsignal3->Clone();
        
        signalhist(hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, signal1, signal2, signal3, lsndy90[i], L1, L2, L3, 0, lsndx90[i], NumBin, expClone1, expClone2, expClone3, back1, back2, back3, back4, back5, back6, A1, A2, A3, nue_true1, nue_true2, nue_true3);
        
        double chisq = ChiSquare(clone1CHI, clone2CHI, clone3CHI, clone4CHI, clone5CHI, clone6CHI, clone7CHI, clone8CHI, clone9CHI, expClone1, expClone2, expClone3,xsec,flux, back1, back2, back3, back4, back5, back6, A1, A2, A3);
        
        lsndsig900.push_back(sqrt(chisq));
    }
    
    TCanvas* canvas4 = new TCanvas("canvas4","SensPlot",200,20,1000,600);
    canvas4->SetLogx();
    
    TGraph* gr2 = new TGraph(lsndyS,&lsndy[0],&lsndsig[0]);
    TGraph* gr3 = new TGraph(lsndyS,&lsndy[0],&lsndsig0[0]);
    gr2->GetXaxis()->SetTitle("#Delta m^{2} (eV^{2})");
    gr2->GetYaxis()->SetTitle("Significance #sqrt{#Delta #chi^{2}}");
    gr2->SetTitle("Sensitivity to 3+1 #nu signal along the LSND 99% CL (sin^{2}(2#theta_{ee})=0.04)");
    gr2->GetXaxis()->SetTitleOffset(1.2);
    gr2->SetLineColor(628);
    gr2->SetMarkerColor(628);

    gr3->SetMarkerStyle(3);
    
    gr2->Draw("AC*");
    gr3->Draw("C* same");
    
    TLegend* lg2 = new TLegend(0.59, 0.58, 0.88, 0.89);
    lg2->AddEntry(gr2, "#nu_{e} Disappearance","p");
    lg2->AddEntry(gr3, "No #nu_{e} Disappearance","p");
    
    lg2->Draw();
    
    canvas4->Modified();
    canvas4->Print("canvas4.pdf");
    
    TCanvas* canvas90 = new TCanvas("canvas90","SensPlot",200,20,1000,600);
    canvas90->SetLogx();
    
    TGraph* grlsnd90 = new TGraph(lsndyS90,&lsndy90[0],&lsndsig90[0]);
    TGraph* grlsnd900 = new TGraph(lsndyS90,&lsndy90[0],&lsndsig900[0]);
    grlsnd90->GetXaxis()->SetTitle("#Delta m^{2} (eV^{2})");
    grlsnd90->GetYaxis()->SetTitle("Significance #sqrt{#Delta #chi^{2}}");
    grlsnd90->SetTitle("Sensitivity to 3+1 #nu signal along the LSND 90% CL (sin^{2}(2#theta_{ee})=0.2)");
    grlsnd90->GetXaxis()->SetTitleOffset(1.2);
    grlsnd90->SetLineColor(596);
    grlsnd90->SetMarkerColor(596);
    
    grlsnd900->SetLineColor(1);
    grlsnd900->SetMarkerColor(1);
    
    grlsnd90->Draw("AC*");
    grlsnd900->Draw("C* same");
    
    TLegend* lg90 = new TLegend(0.59, 0.58, 0.88, 0.89);
    lg90->AddEntry(grlsnd90, "sin^{2}(2 #theta_{ee})=0.2","p");
    lg90->AddEntry(grlsnd900, "No #nu_{e} Disappearance","p");
    
    lg90->Draw();
    
    canvas90->Modified();
    canvas90->Print("canvas90.pdf");
    
    //Set to draw electron neutrino event distributions at the values of delta m^2=1 and sin(2*theta_mue)^2=0.0013 and sin(2*theta_ee)^2 set to the value at beginning of the code. These parameter values should be adjusted to match a desired event distribution.
    sigDraw(expsignal1, expsignal2, expsignal3, 1, 0.0013, theta_ee, hist1, hist2, hist3, hist4, hist5, hist6, hist7, hist8, hist9, signal1, signal2, signal3, L1, L2, L3, NumBin, hs, hs1, hs2, back1, back2, back3, back4, back5, back6, A1, A2, A3, nue_true1, nue_true2, nue_true3,legendPostition,legendPostition2,legendPostition3);

}
