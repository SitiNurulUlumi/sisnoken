  <!-- Full Width Column -->
  <div class="content-wrapper">
      <div class="container">
          <!-- Content Header (Page header) -->
          <section class="content-header">
              <h1>
                  <?= $title ?>
              </h1>
              <ol class="breadcrumb">
                  <li><a href="#"><i class="fa fa-dashboard"></i> <?= $title ?></a></li>
                  <li><a href="#"><?= $subtitle ?></a></li>
              </ol>
          </section>
          <!-- Main content -->
          <section class="content">
              <div class="row">
                  <div class="col-lg-6">
                      <div class="box box-primary">
                          <div class="box-header with-border">
                              <h3 class="box-title"><?= $subtitle ?></h3>
                          </div>
                          <div class="box-body">
                              <form action="#" id="form" class="form-horizontal">
                                  <input type="hidden" value="<?= $idUser ?>" />
                                  <div class="form-body">
                                      <div class="form-group">
                                          <div class="col-md-9">
                                              <input name="estu_id" class="form-control" type="hidden">
                                              <span class="help-block"></span>
                                          </div>
                                      </div>
                                      <div class="form-group">
                                          <label class="control-label col-md-3">Tanda Selar</label>
                                          <div class="col-md-9">
                                              <input name="selar" placeholder="Tanda Selar" class="form-control" type="text">
                                              <span class="help-block"></span>
                                          </div>
                                      </div>
                                      <div class="form-group">
                                          <label class="control-label col-md-3">Nama Kapal</label>
                                          <div class="col-md-9">
                                              <input name="ship_name" placeholder="Nama Kapal" class="form-control" type="text">
                                              <span class="help-block"></span>
                                          </div>
                                      </div>
                                      <div class="form-group">
                                          <label class="control-label col-md-3">Nama Nahkoda</label>
                                          <div class="col-md-9">
                                              <input name="captain" placeholder="Nama Nahkoda" class="form-control" type="text">
                                              <span class="help-block"></span>
                                          </div>
                                      </div>
                                  </div>
                              </form>
                          </div>
                          <!-- /.box-body -->
                          <div class="modal-footer">
                              <button type="button" onclick="goBack()" class="btn btn-social btn-sm btn-warning pull-left"><i class="fa fa-backward"></i> Kembali</button>
                              <button type="button" id="btnSave" onclick="save()" class="btn btn-social btn-sm btn-success"><i class="fa fa-paper-plane"></i> Simpan</button>
                          </div>
                      </div>
                      <!-- /.box -->
                  </div>
              </div>

          </section>
          <!-- /.content -->

      </div>
      <!-- /.container -->
  </div>