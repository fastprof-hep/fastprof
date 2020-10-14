{
auto model = HftModelBuilder::Create("inputs/model.dat");
auto state = model->CurrentState();
data = model->GenerateEvents();
data->SetName("obsData");
model->LoadState(state);
model->ExportWorkspace("outputs/model.root", "modelWS", "xs", data);
}
