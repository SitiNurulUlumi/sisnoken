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
                  <div class="col-lg-12">
                      <div class="box box-primary">
                          <div class="box-header with-border">
                              <h3 class="box-title"><?= $subtitle ?></h3>
                          </div>
                          <div class="box-body">
                              <button class="btn btn-social btn-sm btn-success" onclick="add()"><i class="glyphicon glyphicon-plus"></i> Tambah Data</button>
                              <button class="btn btn-social btn-sm btn-warning" onclick="reloadTable()"><i class="glyphicon glyphicon-refresh"></i> Reload Data</button>
                              <button id="deleteList" class="btn btn-social btn-sm btn-danger" style="display: none;" onclick="deleteList()"><i class="glyphicon glyphicon-trash"></i> Hapus Masal</button>
                              <hr>
                              <div class="table-responsive">
                                  <table id="table" class="table table-bordered table-striped" width="100%">
                                      <thead>
                                          <tr>
                                              <th><input type="checkbox" id="check-all"></th>
                                              <th>Aksi</th>
                                              <th>Nama Lengkap</th>
                                              <th>Username</th>
                                          </tr>
                                      </thead>
                                      <tbody>
                                      </tbody>
                                      <tfoot>
                                          <tr>
                                              <th></th>
                                              <th>Aksi</th>
                                              <th>Nama Lengkap</th>
                                              <th>Username</th>
                                          </tr>
                                      </tfoot>
                                  </table>
                              </div>
                          </div>
                          <!-- /.box-body -->
                      </div>
                      <!-- /.box -->
                  </div>
              </div>
      </div> -->
  </div>

  </section>
  <!-- /.content -->

  </div>
  <!-- /.container -->
  </div>
  <!-- Bootstrap modal -->
  <div class="modal fade" id="modal_form" role="dialog">
      <div class="modal-dialog">
          <div class="modal-content">
              <div class="modal-header">
                  <h3 class="modal-title">Form</h3>
              </div>
              <div class="modal-body form">
                  <form action="#" id="form" class="form-horizontal">
                      <input type="hidden" value="" name="id" />
                      <div class="form-body">
                          <div class="form-group" id="fullname">
                              <label class="control-label col-md-3">Nama Lengkap</label>
                              <div class="col-md-9">
                                  <input name="fullname" placeholder="Nama Lengkap" class="form-control" type="text">
                                  <span class="help-block"></span>
                              </div>
                          </div>
                          <div class="form-group" id="email">
                              <label class="control-label col-md-3">Username</label>
                              <div class="col-md-9">
                                  <input name="email" placeholder="Username" class="form-control" type="text">
                                  <span class="help-block"></span>
                              </div>
                          </div>
                          <div class="form-group" id="password">
                              <label class="control-label col-md-3">Password</label>
                              <div class="col-md-9">
                                  <input type="password" name="password" placeholder="Password" class="form-control" type="text">
                                  <span class="help-block"></span>
                              </div>
                          </div>
                          <div class="form-group" id="repassword">
                              <label class="control-label col-md-3">Re Password</label>
                              <div class="col-md-9">
                                  <input type="password" name="repassword" placeholder="Masukan Ulang Password" class="form-control" type="text">
                                  <span class="help-block"></span>
                              </div>
                          </div>
                          <div class="form-group" id="plat_nomor">
                              <label class="control-label col-md-3">Nomor Kendaraan</label>
                              <div class="col-md-9">
                                  <input name="plat_nomor" placeholder="Nomor Kendaraan" class="form-control" type="text">
                                  <span class="help-block"></span>
                              </div>
                          </div>
                          <!-- <div class="form-group" id="ship_name">
                              <label class="control-label col-md-3">Nama Kapal</label>
                              <div class="col-md-9">
                                  <input name="ship_name" placeholder="Nama Kapal" class="form-control" type="text">
                                  <span class="help-block"></span>
                              </div>
                          </div>
                          <div class="form-group" id="captain">
                              <label class="control-label col-md-3">Nama Nahkoda</label>
                              <div class="col-md-9">
                                  <input name="captain" placeholder="Nama Nahkoda" class="form-control" type="text">
                                  <span class="help-block"></span>
                              </div>
                          </div> -->
                      </div>
                  </form>
              </div>
              <div class="modal-footer">
                  <button type="button" onclick="close_modal()" class="btn btn-social btn-sm btn-danger" data-dismiss="modal"><i class="fa fa-times"></i> Batal</button>
                  <button type="button" id="btnSave" onclick="save()" class="btn btn-social btn-sm btn-success"><i class="fa fa-paper-plane"></i> Simpan</button>
              </div>
          </div><!-- /.modal-content -->
      </div><!-- /.modal-dialog -->
  </div><!-- /.modal -->