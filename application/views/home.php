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
        <?php
        if ($this->session->userdata('is_logged_in') && $this->session->userdata('role') != 3) {
        ?>
          <div class="row">
            <div class="col-lg-3 col-xs-6">
              <!-- small box -->
              <div class="small-box bg-aqua">
                <div class="inner">
                  <h3><?= $history ? $history : '0' ?></h3>

                  <p>Total Kendaraan Masuk</p>
                </div>
                <div class="icon">
                  <i class="fa fa-retweet"></i>
                </div>
              </div>
            </div>
            <!-- ./col -->
            <div class="col-lg-3 col-xs-6">
              <div class="small-box bg-yellow">
                <div class="inner">
                  <h3><?= $historykeluar ? $historykeluar : '0' ?></h3>

                  <p>Total Kendaraan Keluar</p>
                </div>
                <div class="icon">
                  <i class="fa fa-list-ol"></i>
                </div>
              </div>
            </div>
          </div>
        <?php } ?>


        <!-- /.box -->
      </section>
      <!-- /.content -->
    </div>
    <!-- /.container -->
  </div>