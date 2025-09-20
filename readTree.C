
void readTree(const char *input, const char *output = "pfrich.root",int momentum=0,std::string particleName=""){
  auto finput  = new TFile(input);
  auto geometry = dynamic_cast<CherenkovDetectorCollection*>(finput->Get("CherenkovDetectorCollection"));
  TTree *t = dynamic_cast<TTree*>(finput->Get("t")); 
  auto event = new CherenkovEvent();
  t->SetBranchAddress("e", &event);

  //Output file name
  TFile *ofile = TFile::Open(output, "recreate");
  TTree *ptree = new TTree("Events","Cherenkov Events"); 

  int pdgID;
  double posX,posY, posZ;
  double momX, momY, momZ;
  double eta, pT, theta, phi, time;
  ptree->Branch("pdgID", &pdgID, "pdgID/I");
  ptree->Branch("posX", &posX, "posX/D");
  ptree->Branch("posY", &posY, "posY/D");
  ptree->Branch("posZ", &posZ, "posZ/D");
  ptree->Branch("momX", &momX, "momX/D");
  ptree->Branch("momY", &momY, "momY/D");
  ptree->Branch("momZ", &momZ, "momZ/D");
  ptree->Branch("pT", &pT, "pT/D");
  ptree->Branch("eta", &eta, "eta/D");
  ptree->Branch("theta", &theta, "theta/D");
  ptree->Branch("phi", &phi, "phi/D");
  ptree->Branch("time", &time, "time/D");
  std::string histname=particleName+"_"+std::to_string(momentum)+"GeV_XY"; 
  auto hxy = new TH2D(histname.c_str(), "", 650, -650., 650., 650, -650.0, 650.);

  int parentID=-999;
  int nEvents = t->GetEntries();
  for(unsigned ev=0; ev<nEvents; ev++){
    t->GetEntry(ev);
    for(auto particle: event->ChargedParticles()){
      parentID=particle->GetPDG();
      for(auto rhistory: particle->GetRadiatorHistory()){
	auto history  = particle->GetHistory(rhistory);
	for(auto photon: history->Photons()){
	  if(!photon->WasDetected()) continue;
	  TVector3 phx = photon->GetDetectionPosition();
	  TVector3 mom = photon->GetVertexParentMomentum();
	  hxy->Fill(phx.X(), phx.Y());
	  posX=phx.X();
          posY=phx.Y();
          posZ=phx.Z();
	  momX=mom.X();
          momY=mom.Y();
          momZ=mom.Z();
          theta=phx.Theta();
          phi=phx.Phi();
          pT=phx.Pt();
          pT=phx.Pt();
          eta=phx.PseudoRapidity();
	  time = photon->GetDetectionTime();
	  pdgID = parentID;
	} //for photon
      } //for rhistory
    } //for particle
    ptree->Fill();
  } //for ev

  //gStyle->SetOptStat(0);
  auto cv = new TCanvas("cv", "", 1000, 1000);
  hxy->GetXaxis()->SetTitle("Sensor plane X, [mm]");
  hxy->GetYaxis()->SetTitle("Sensor plane Y, [mm]");
  hxy->Draw("COL");
  hxy->Write();
  ofile->Write();
  ofile->Close();
} // pfrich()
