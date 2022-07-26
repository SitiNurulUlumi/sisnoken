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
                              <form action="" method="POST" enctype="multipart/form-data" class="form-horizontal">
                                  <input type="hidden" value="" name="id" />
                                  <div class="form-body">

                                      <div class="form-group" id="repassword">
                                          <label class="control-label col-md-3">Foto Plat</label>
                                          <div class="col-md-9">
                                              <input type="file" name="plat" id="plat" class="form-control" accept="image/png, image/jpeg, image/jpg, image/gif" required>
                                              <?php if (isset($error)) : ?>
                                                  <div class="help-block"><?= $error ?></div>
                                                  <!-- <span class="help-block"></span> -->
                                              <?php endif; ?>
                                          </div>
                                      </div>
                                      <div class="col-md-12" id="form-footer">
                                          <button type="submit" class="btn btn-social btn-sm btn-success pull-right"><i class="fa fa-paper-plane"></i> Simpan</button>
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