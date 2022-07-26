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
                                  <input type="hidden" value="" name="id" />
                                  <div class="form-body">
                                      <div class="form-group">
                                          <label class="control-label col-md-3">Nama Lengkap</label>
                                          <div class="col-md-9">
                                              <input value="<?= $profile->fullname ?>" class="form-control" type="text" disabled>
                                          </div>
                                      </div>
                                      <div class="form-group">
                                          <label class="control-label col-md-3">Username</label>
                                          <div class="col-md-9">
                                              <input value="<?= $profile->email ?>" class="form-control" type="text" disabled>
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
                                      <div class="form-group pull-left">
                                          <small class="control-label col-md-12">*isi password dan repassword untuk merubah password akun anda.</small>
                                      </div>
                                      <div class="col-md-12" id="form-footer">
                                          <button type="button" id="btnSave" onclick="save()" class="btn btn-social btn-sm btn-success pull-right"><i class="fa fa-paper-plane"></i> Simpan</button>
                                      </div>
                                  </div>
                              </form>
                          </div>
                          <!-- /.box-body -->
                      </div>
                      <!-- /.box -->
                  </div>
              </div>

          </section>
          <!-- /.content -->

      </div>
      <!-- /.container -->
  </div>