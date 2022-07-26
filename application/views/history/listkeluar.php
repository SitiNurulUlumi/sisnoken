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
                              <button class="btn btn-social btn-sm btn-warning" onclick="reloadTable()"><i class="glyphicon glyphicon-refresh"></i> Reload Data</button>
                              <a class="btn btn-social btn-sm btn-success" href="<?= base_url() ?>images/keluar/"><i class="glyphicon glyphicon-plus"></i> Tambah Foto</a>
                              <hr>
                              <div class="table-responsive">
                                  <table id="table" class="table table-bordered table-striped" width="100%">
                                      <thead>
                                          <tr>
                                              <th>No</th>
                                              <th>Plat Nomor</th>
                                              <th>Keluar</th>
                                          </tr>
                                      </thead>
                                      <tbody>
                                      </tbody>
                                      <tfoot>
                                          <tr>
                                              <th>No</th>
                                              <th>Plat Nomor</th>
                                              <th>Keluar</th>
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

          </section>
          <!-- /.content -->

      </div>
      <!-- /.container -->
  </div>