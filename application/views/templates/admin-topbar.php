<header class="main-header">
  <nav class="navbar navbar-static-top">
    <div class="container">
      <div class="navbar-header">
        <a href="<?= base_url() ?>" class="navbar-brand"><b>SISNOKEN</b> APP</a>
        <button type="button" class="navbar-toggle collapsed" data-toggle="collapse" data-target="#navbar-collapse">
          <i class="fa fa-bars"></i>
        </button>
      </div>

      <!-- Collect the nav links, forms, and other content for toggling -->
      <div class="collapse navbar-collapse pull-left" id="navbar-collapse">
        <ul class="nav navbar-nav">
          <li class="<?= $title == 'Beranda' ? 'active' : '' ?>"><a href="<?= base_url() ?>home/">Beranda <span class="sr-only">(current)</span></a></li>
          <?php if ($this->session->userdata('role') == 1 || $this->session->userdata('role') == 2) : ?>
            <li class="dropdown <?= $title == 'Data Pemilik Kendaraan' || $title == 'Data Kendaraan'  || $title == 'Data Admin' ? 'active' : '' ?>">
              <a href="#" class="dropdown-toggle" data-toggle="dropdown"><?= $this->session->userdata('role') == '1' ? 'Menu Administrator' : 'Menu Admin' ?> <span class="caret"></span></a>
              <ul class="dropdown-menu" role="menu">
                <?php if ($this->session->userdata('role') == 1) : ?>
                  <li class="<?= $title == 'Data Admin' ? 'active' : '' ?>"><a href="<?= base_url() ?>admins/list/">Data Admin</a></li>
                  <?php endif; ?>
                  <?php if ($this->session->userdata('role') == 2) : ?>
                    <!-- <li class="<?= $title == 'Data Pemilik Kendaraan' ? 'active' : '' ?>"><a href="<?= base_url() ?>users/list/">Data Pemilik Kendaraan</a></li> -->
                    <!-- <li class="<?= $title == 'Data Kendaraan' ? 'active' : '' ?>"><a href="<?= base_url() ?>plats/list/">Data Kendaraan</a></li> -->
                    <li class="<?= $title == 'History Kendaraan Masuk' ? 'active' : '' ?>"><a href="<?= base_url() ?>history/">Data History Masuk</a></li>
                    <li class="<?= $title == 'History Kendaraan Keluar' ? 'active' : '' ?>"><a href="<?= base_url() ?>historykeluar/">Data History Keluar</a></li>
                    <!-- <li class="<?= $title == 'Data Foto Nomor Plat' ? 'active' : '' ?>"><a href="<?= base_url() ?>images/list/">Data Foto Nomor Plat</a></li> -->
                    <?php endif; ?>
                  </ul>
                </li>
                <?php endif; ?>
                <?php if ($this->session->userdata('role') == 1) : ?>
                  <li class="<?= $title == 'Log' ? 'active' : '' ?>"><a href="<?= base_url() ?>logs/">Log Aktivitas <span class="sr-only">(current)</span></a></li>
                  <?php endif; ?>
                  <li class="<?= $title == 'Chat Bot' ? 'active' : '' ?>"><a href="http://127.0.0.1:5000/">Chat Bot <span class="sr-only">(current)</span></a></li>
                  <?php if ($this->session->userdata('is_logged_in')) : ?>
            <li class="dropdown">
              <a href="#" class="dropdown-toggle" data-toggle="dropdown">Akun <span class="caret"></span></a>
              <ul class="dropdown-menu" role="menu">
                <li class="<?= $title == 'Profile' ? 'active' : '' ?>"><a href="<?= base_url() ?>profile">Profil</a></li>
                <li class="divider"></li>
                <li><a href="<?= base_url() ?>auth/logout/">Keluar</a></li>
              </ul>
            </li>
          <?php else : ?>
            <li class="<?= $title == 'Login' ? 'active' : '' ?>"><a href="<?= base_url() ?>auth/login/">Masuk</a></li>
          <?php endif; ?>
        </ul>
      </div>
      <!-- /.navbar-collapse -->
      <!-- Navbar Right Menu -->
      <?php if ($this->session->userdata('is_logged_in')) : ?>
        <div class="navbar-custom-menu">
          <ul class="nav navbar-nav">
            <!-- Messages: style can be found in dropdown.less-->
            <!-- User Account Menu -->
            <li class="dropdown user user-menu active">
              <!-- Menu Toggle Button -->
              <a class="active">
                <!-- The user image in the navbar-->
                <!-- hidden-xs hides the username on small devices so only the image appears. -->
                <span>Halo, <?= $this->session->userdata('email') ?></span>
              </a>
            </li>
          </ul>
        </div>
      <?php endif; ?>
      <!-- /.navbar-custom-menu -->
    </div>
    <!-- /.container-fluid -->
  </nav>
</header>